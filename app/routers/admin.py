from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import APIKeyHeader
from app.core.config import settings
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import FoodLog, SportActivity, User, WeightLog
from datetime import datetime, timezone
from app.services.summary import upsert_daily_summary, backfill_all_summaries
from sqlalchemy import func
from pydantic import BaseModel
from typing import List
import uuid as _uuid
from sqlalchemy import text as _text
import threading
import time

router = APIRouter()

# In-process status for background jobs
POPULATE_STATUS: dict = {"state": "idle", "error": None, "counts": {}}

# Define API key security scheme
api_key_header = APIKeyHeader(name="X-API-Key")


def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == settings.ADMIN_API_KEY:
        return api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
    )


@router.post("/retrain", status_code=status.HTTP_200_OK)
async def retrain_model(api_key: str = Depends(get_api_key)):
    # Training requires PyTorch and is not available in the production container.
    # Provide a clear response without importing heavy training code.
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Retraining is disabled in this deployment (NumPy-only inference). Train offline and upload models/recurrent_model_np.npz and models/best_params.json.",
    )


@router.post("/backfill-daily-summaries", status_code=status.HTTP_200_OK)
def backfill_daily_summaries(
    api_key: str = Depends(get_api_key), db: Session = Depends(get_db)
):
    # Prefer dummy public user if present; else compute global summaries (user_id None)
    dummy_user = db.query(User).filter(User.email == "dummy@example.com").first()
    user_id = dummy_user.id if dummy_user else None

    # Collect distinct dates from food_logs and sport_activities
    food_dates = db.query(func.date(FoodLog.logged_at)).distinct().all()
    sport_dates = db.query(func.date(SportActivity.logged_at)).distinct().all()
    dates = {d[0] for d in food_dates} | {d[0] for d in sport_dates}

    updated = backfill_all_summaries(db, user_id)
    return {"updated_days": updated}


class WeightImportItem(BaseModel):
    date: str
    weight_kg: float


@router.post("/weights/import", status_code=status.HTTP_200_OK)
def import_weights(
    items: List[WeightImportItem],
    api_key: str = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    # Ensure a dummy public user exists (for public plots aggregation)
    dummy_user = db.query(User).filter(User.email == "dummy@example.com").first()
    if not dummy_user:
        # Insert via SQL to satisfy schema differences (username column)
        user_id = str(_uuid.uuid4())
        db.execute(
            _text(
                """
                INSERT INTO users (id, email, hashed_password, username)
                VALUES (:id, :email, :hashed_password, :username)
                """
            ),
            {
                "id": user_id,
                "email": "dummy@example.com",
                "hashed_password": "imported",
                "username": "dummy",
            },
        )
        db.commit()
        dummy_user = db.query(User).filter(User.id == user_id).first()

    # Upsert weights per day for the dummy user
    from datetime import datetime as _dt, timezone as _tz

    inserted = 0
    for it in items:
        try:
            d = _dt.strptime(it.date, "%Y-%m-%d")
        except ValueError:
            # Try alternative formats
            d = _dt.fromisoformat(it.date[0:10])
        day_dt = _dt(d.year, d.month, d.day, tzinfo=_tz.utc)
        # Delete existing for that day/user (idempotent)
        db.query(WeightLog).filter(
            WeightLog.user_id == dummy_user.id,
            func.date(WeightLog.logged_at) == day_dt.date(),
        ).delete()
        wl = WeightLog(
            user_id=dummy_user.id,
            weight_kg=float(it.weight_kg),
            logged_at=day_dt,
            logged_date=day_dt.date(),
        )
        db.add(wl)
        inserted += 1
        if inserted % 200 == 0:
            db.commit()
    db.commit()
    return {"inserted": inserted}


@router.post("/populate", status_code=status.HTTP_200_OK)
def populate_all(api_key: str = Depends(get_api_key)):
    """Populate foods, food_logs, and sport_activities from packaged CSVs on the server."""
    try:
        # Import lazily to avoid heavy deps at router import time
        from app.db.populate_db import (
            populate_food_table,
            populate_food_log_table,
            populate_sport_activities_table,
        )

        populate_food_table()
        populate_food_log_table()
        populate_sport_activities_table()
        # Capture counts
        from sqlalchemy import text as _text
        from app.db.database import SessionLocal as _Sess

        sess = _Sess()
        try:
            foods = sess.execute(_text("SELECT COUNT(*) FROM foods")).scalar() or 0
            logs = sess.execute(_text("SELECT COUNT(*) FROM food_logs")).scalar() or 0
            sports = (
                sess.execute(_text("SELECT COUNT(*) FROM sport_activities")).scalar()
                or 0
            )
        finally:
            sess.close()
        return {
            "foods": int(foods),
            "food_logs": int(logs),
            "sport_activities": int(sports),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Populate failed: {e}")


# --- Database backup endpoints and scheduler ---

from typing import Optional

_BACKUP_THREAD: Optional[threading.Thread] = None
_BACKUP_SCHEDULER_RUNNING: bool = False


def _backup_loop_daily() -> None:
    global _BACKUP_SCHEDULER_RUNNING
    _BACKUP_SCHEDULER_RUNNING = True
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    from app.services.backup import perform_backup

    while True:
        try:
            now = _dt.now(_tz.utc)
            next_midnight = (now + _td(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            sleep_s = max(1.0, (next_midnight - now).total_seconds())
            time.sleep(sleep_s)
            try:
                _ = perform_backup()
            except Exception:
                pass
        except Exception:
            time.sleep(60.0)


def start_backup_scheduler_internal() -> dict:
    global _BACKUP_THREAD, _BACKUP_SCHEDULER_RUNNING
    if _BACKUP_THREAD and _BACKUP_THREAD.is_alive():
        return {"running": True}
    try:
        t = threading.Thread(target=_backup_loop_daily, daemon=True)
        t.start()
        _BACKUP_THREAD = t
        return {"running": True}
    except Exception as e:
        raise RuntimeError(f"Failed to start scheduler: {e}")


@router.post("/backup/run", status_code=status.HTTP_200_OK)
def run_backup_now(api_key: str = Depends(get_api_key)):
    from app.services.backup import perform_backup

    try:
        result = perform_backup()
        return {"ok": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup failed: {e}")


@router.post("/backup/scheduler/start", status_code=status.HTTP_200_OK)
def start_backup_scheduler(api_key: str = Depends(get_api_key)):
    try:
        return start_backup_scheduler_internal()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/backup/scheduler/status", status_code=status.HTTP_200_OK)
def backup_scheduler_status(api_key: str = Depends(get_api_key)):
    global _BACKUP_THREAD
    return {
        "running": bool(_BACKUP_THREAD and _BACKUP_THREAD.is_alive()),
    }


def _populate_job_impl() -> None:
    global POPULATE_STATUS
    POPULATE_STATUS = {"state": "running", "error": None, "counts": {}}
    try:
        from app.db.populate_db import (
            populate_food_table,
            populate_food_log_table,
            populate_sport_activities_table,
        )

        populate_food_table()
        populate_food_log_table()
        populate_sport_activities_table()
        # Summarize counts
        from sqlalchemy import text as _text
        from app.db.database import SessionLocal as _Sess

        sess = _Sess()
        try:
            foods = sess.execute(_text("SELECT COUNT(*) FROM foods")).scalar() or 0
            logs = sess.execute(_text("SELECT COUNT(*) FROM food_logs")).scalar() or 0
            sports = (
                sess.execute(_text("SELECT COUNT(*) FROM sport_activities")).scalar()
                or 0
            )
        finally:
            sess.close()
        POPULATE_STATUS = {
            "state": "done",
            "error": None,
            "counts": {
                "foods": int(foods),
                "food_logs": int(logs),
                "sport_activities": int(sports),
            },
        }
    except Exception as exc:
        POPULATE_STATUS = {"state": "error", "error": str(exc), "counts": {}}


@router.post("/populate-async", status_code=status.HTTP_202_ACCEPTED)
def populate_all_async(
    background_tasks: BackgroundTasks, api_key: str = Depends(get_api_key)
):
    """Trigger population in the background to avoid request timeouts on Render free tier."""
    background_tasks.add_task(_populate_job_impl)
    return {"accepted": True}


@router.get("/populate/status", status_code=status.HTTP_200_OK)
def populate_status(api_key: str = Depends(get_api_key)):
    return POPULATE_STATUS


# ---- Prediction parity (NumPy vs Torch) ----


@router.get("/predict/parity", status_code=status.HTTP_200_OK)
def predict_parity(api_key: str = Depends(get_api_key)):
    from app.main import prediction_service, SessionLocal
    from sqlalchemy import func as _func
    import pandas as _pd

    # Build recent features from DB
    db = SessionLocal()
    try:
        dfl = _func.coalesce(FoodLog.logged_date, _func.date(FoodLog.logged_at))
        food_rows = (
            db.query(
                dfl.label("date"),
                _func.coalesce(_func.sum(FoodLog.calories), 0.0).label("calories"),
                _func.coalesce(_func.sum(FoodLog.carbs), 0.0).label("carbs"),
            )
            .group_by(dfl)
            .order_by(dfl)
            .all()
        )
        dsa = _func.coalesce(
            SportActivity.logged_date, _func.date(SportActivity.logged_at)
        )
        sport_rows = (
            db.query(
                dsa.label("date"),
                _func.coalesce(_func.sum(SportActivity.calories_expended), 0.0).label(
                    "sport"
                ),
            )
            .group_by(dsa)
            .order_by(dsa)
            .all()
        )
    finally:
        db.close()
    dates = sorted({r.date for r in food_rows} | {r.date for r in sport_rows})
    dates = dates[-30:] if len(dates) > 30 else dates
    food_by_date = {r.date: r for r in food_rows}
    sport_by_date = {r.date: r for r in sport_rows}
    records: list[dict] = []
    for d in dates:
        fr = food_by_date.get(d)
        sr = sport_by_date.get(d)
        records.append(
            {
                "date": d,
                "calories": float(getattr(fr, "calories", 0.0) or 0.0),
                "carbs": float(getattr(fr, "carbs", 0.0) or 0.0),
                "sugar": 0.0,
                "sel": 0.0,
                "alcool": 0.0,
                "water": 0.0,
                "sport": float(getattr(sr, "sport", 0.0) or 0.0),
                "pds": 0.0,
            }
        )
    features_df = _pd.DataFrame(records)

    outs_numpy = prediction_service.predict_from_features(
        features_df.copy(), backend="numpy"
    )
    outs_torch = prediction_service.predict_from_features(
        features_df.copy(), backend="torch"
    )

    import numpy as _np

    def _summ(outs: dict) -> dict:
        w = _np.asarray(outs.get("predicted_adjusted_weight", []), dtype=float)
        m = _np.asarray(outs.get("base_metabolism_kcal", []), dtype=float)
        return {
            "len": int(w.size),
            "w_minmax": [
                float(_np.nanmin(w)) if w.size else None,
                float(_np.nanmax(w)) if w.size else None,
            ],
            "m_minmax": [
                float(_np.nanmin(m)) if m.size else None,
                float(_np.nanmax(m)) if m.size else None,
            ],
            "w_tail": [float(v) for v in w[-3:]] if w.size else [],
            "m_tail": [float(v) for v in m[-3:]] if m.size else [],
        }

    return {
        "normalization": prediction_service.normalization_stats,
        "numpy": _summ(outs_numpy),
        "torch": _summ(outs_torch),
    }


# ---- Prediction features preview (diagnostics) ----


@router.get("/predict/features", status_code=status.HTTP_200_OK)
def predict_features_preview(n: int = 10, api_key: str = Depends(get_api_key)):
    """Return the last n rows of features used by the server for predictions.

    Helps diagnose discrepancies between CSV and DB-derived features.
    """
    try:
        from app.features.builder import build_prediction_features_from_db

        df = build_prediction_features_from_db()
        if df is None or df.empty:
            return {"rows": 0}
        tail = df.tail(max(1, int(n)))
        # Ensure JSON-serializable columns only
        cols = [
            "date",
            "calories",
            "carbs",
            "sugar",
            "sel",
            "alcool",
            "water",
            "sport",
            "pds",
        ]
        cols = [c for c in cols if c in tail.columns]
        data = {c: [float(v) if c != "date" else str(v) for v in tail[c].tolist()] for c in cols}
        return {"rows": int(len(tail)), "features_tail": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build features: {e}")
