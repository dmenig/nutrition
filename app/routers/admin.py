from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from app.core.config import settings
from app.jobs.retrain_model import retrain_model_job
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import FoodLog, SportActivity, User
from datetime import datetime, timezone
from app.services.summary import upsert_daily_summary, backfill_all_summaries
from app.db.populate_db import (
    populate_food_table,
    populate_food_log_table,
    populate_sport_activities_table,
    verify_population,
)
from sqlalchemy import func

router = APIRouter()

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
    try:
        retrain_model_job()
        return {"message": "Model retraining initiated successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model retraining failed: {e}",
        )


@router.post("/backfill-daily-summaries", status_code=status.HTTP_200_OK)
def backfill_daily_summaries(api_key: str = Depends(get_api_key), db: Session = Depends(get_db)):
    # Prefer dummy public user if present; else compute global summaries (user_id None)
    dummy_user = db.query(User).filter(User.email == "dummy@example.com").first()
    user_id = dummy_user.id if dummy_user else None

    # Collect distinct dates from food_logs and sport_activities
    food_dates = db.query(func.date(FoodLog.logged_at)).distinct().all()
    sport_dates = db.query(func.date(SportActivity.logged_at)).distinct().all()
    dates = {d[0] for d in food_dates} | {d[0] for d in sport_dates}

    updated = backfill_all_summaries(db, user_id)
    return {"updated_days": updated}


@router.post("/populate", status_code=status.HTTP_200_OK)
def populate_all(api_key: str = Depends(get_api_key)):
    """Populate foods, food_logs, and sport_activities from packaged CSVs on the server."""
    try:
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
            sports = sess.execute(_text("SELECT COUNT(*) FROM sport_activities")).scalar() or 0
        finally:
            sess.close()
        return {"foods": int(foods), "food_logs": int(logs), "sport_activities": int(sports)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Populate failed: {e}")
