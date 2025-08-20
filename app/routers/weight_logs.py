from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from datetime import datetime as _dt, timezone as _tz, date as _date

from app.db.database import get_db
from app.core.auth import get_current_user
from app.db.models import WeightLog
from app.schemas import WeightLogCreate, WeightLogOut
from app.services.summary import rebuild_predictions_cache_async


router = APIRouter()


def _day_start_utc(dt: _dt) -> _dt:
    return _dt(dt.year, dt.month, dt.day, tzinfo=_tz.utc)


@router.post("/api/v1/weights", response_model=WeightLogOut, status_code=status.HTTP_201_CREATED)
def upsert_weight_for_day(
    payload: WeightLogCreate,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    # Normalize to day start in UTC and set logged_date
    if payload.logged_at.tzinfo is None:
        logged_at = payload.logged_at.replace(tzinfo=_tz.utc)
    else:
        logged_at = payload.logged_at.astimezone(_tz.utc)
    day_dt = _day_start_utc(logged_at)

    # Delete existing entry for this user/day (idempotent upsert)
    db.query(WeightLog).filter(
        WeightLog.user_id == current_user.id,
        func.coalesce(WeightLog.logged_date, func.date(WeightLog.logged_at)) == day_dt.date(),
    ).delete()

    wl = WeightLog(
        user_id=current_user.id,
        weight_kg=float(payload.weight_kg),
        logged_at=day_dt,
        logged_date=day_dt.date(),
    )
    db.add(wl)
    db.commit()
    db.refresh(wl)

    # If backfilling a past date, rebuild prediction/plot caches
    try:
        today_utc = _dt.now(tz=_tz.utc).date()
        if wl.logged_at.date() < today_utc:
            rebuild_predictions_cache_async()
    except Exception:
        pass
    return wl


@router.get("/api/v1/weights", response_model=WeightLogOut)
def get_weight_for_day(
    date: _date = Query(..., description="Date in YYYY-MM-DD"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    # Lookup by logged_date or date(logged_at)
    day_start = _dt(date.year, date.month, date.day, tzinfo=_tz.utc)
    wl = (
        db.query(WeightLog)
        .filter(
            WeightLog.user_id == current_user.id,
            func.coalesce(WeightLog.logged_date, func.date(WeightLog.logged_at)) == day_start.date(),
        )
        .order_by(WeightLog.logged_at.desc())
        .first()
    )
    if wl is None:
        raise HTTPException(status_code=404, detail="No weight logged for this date")
    return wl


class _WeightBulkItem(WeightLogCreate):
    pass


@router.post("/api/v1/weights/bulk", response_model=int)
def upsert_weights_bulk(
    items: List[_WeightBulkItem],
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    inserted = 0
    for it in items:
        if it.logged_at.tzinfo is None:
            logged_at = it.logged_at.replace(tzinfo=_tz.utc)
        else:
            logged_at = it.logged_at.astimezone(_tz.utc)
        day_dt = _day_start_utc(logged_at)
        db.query(WeightLog).filter(
            WeightLog.user_id == current_user.id,
            func.coalesce(WeightLog.logged_date, func.date(WeightLog.logged_at)) == day_dt.date(),
        ).delete()
        wl = WeightLog(
            user_id=current_user.id,
            weight_kg=float(it.weight_kg),
            logged_at=day_dt,
            logged_date=day_dt.date(),
        )
        db.add(wl)
        inserted += 1
        if inserted % 200 == 0:
            db.commit()
    db.commit()
    try:
        rebuild_predictions_cache_async()
    except Exception:
        pass
    return inserted


