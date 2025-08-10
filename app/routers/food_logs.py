from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, date, timezone, timedelta
from sqlalchemy import func  # Import func for aggregation

from app.db.models import FoodLog, User
from app.schemas import (
    FoodLogCreate,
    FoodLogUpdate,
    FoodLogOut,
    DailyFoodLogSummary,
)  # Import DailyFoodLogSummary
from app.core.auth import get_current_user
from app.db.database import get_db
from app.services.summary import upsert_daily_summary

router = APIRouter()


@router.get("/api/v1/logs/summary", response_model=DailyFoodLogSummary)
def get_daily_food_log_summary(
    date: date = Query(..., description="Date in YYYY-MM-DD format"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    day_start = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    summary = (
        db.query(
            func.coalesce(func.sum(FoodLog.calories), 0).label("calories"),
            func.coalesce(func.sum(FoodLog.protein), 0).label("protein_g"),
            func.coalesce(func.sum(FoodLog.carbs), 0).label("carbs_g"),
            func.coalesce(func.sum(FoodLog.fat), 0).label("fat_g"),
        )
        .filter(
            FoodLog.user_id == current_user.id,
            FoodLog.logged_at >= day_start,
            FoodLog.logged_at < day_end,
        )
        .first()
    )

    if not summary:
        # Return zeros if no logs found for the date
        return DailyFoodLogSummary(calories=0.0, protein_g=0.0, carbs_g=0.0, fat_g=0.0)

    return DailyFoodLogSummary(**summary._asdict())


@router.post("/api/v1/logs", response_model=FoodLogOut)
def create_food_log(
    food_log: FoodLogCreate,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    db_food_log = FoodLog(**food_log.dict(), user_id=current_user.id)
    db.add(db_food_log)
    db.commit()
    db.refresh(db_food_log)
    # Update daily summary for the specific day
    upsert_daily_summary(db, db_food_log.logged_at, current_user.id)
    return db_food_log


@router.get("/api/v1/logs", response_model=List[FoodLogOut])
def get_food_logs_for_date(
    date: date = Query(..., description="Date in YYYY-MM-DD format"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    day_start = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    food_logs = (
        db.query(FoodLog)
        .filter(
            FoodLog.user_id == current_user.id,
            FoodLog.logged_at >= day_start,
            FoodLog.logged_at < day_end,
        )
        .order_by(FoodLog.logged_at.asc())
        .all()
    )
    return food_logs


@router.get("/api/v1/logs/public", response_model=List[FoodLogOut])
def get_food_logs_for_date_public(
    date: date = Query(..., description="Date in YYYY-MM-DD format"),
    db: Session = Depends(get_db),
):
    day_start = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    dummy_user = db.query(User).filter(User.email == "dummy@example.com").first()

    query = db.query(FoodLog).filter(
        FoodLog.logged_at >= day_start,
        FoodLog.logged_at < day_end,
    )
    if dummy_user:
        query = query.filter(FoodLog.user_id == dummy_user.id)

    food_logs = query.order_by(FoodLog.logged_at.asc()).all()
    return food_logs


@router.get("/api/v1/logs/summary/public", response_model=DailyFoodLogSummary)
def get_daily_food_log_summary_public(
    date: date = Query(..., description="Date in YYYY-MM-DD format"),
    db: Session = Depends(get_db),
):
    day_start = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    dummy_user = db.query(User).filter(User.email == "dummy@example.com").first()

    base_query = db.query(
        func.coalesce(func.sum(FoodLog.calories), 0).label("calories"),
        func.coalesce(func.sum(FoodLog.protein), 0).label("protein_g"),
        func.coalesce(func.sum(FoodLog.carbs), 0).label("carbs_g"),
        func.coalesce(func.sum(FoodLog.fat), 0).label("fat_g"),
    ).filter(
        FoodLog.logged_at >= day_start,
        FoodLog.logged_at < day_end,
    )

    if dummy_user:
        base_query = base_query.filter(FoodLog.user_id == dummy_user.id)

    summary = base_query.first()

    if not summary:
        return DailyFoodLogSummary(calories=0.0, protein_g=0.0, carbs_g=0.0, fat_g=0.0)

    return DailyFoodLogSummary(**summary._asdict())


@router.put("/api/v1/logs/{log_id}", response_model=FoodLogOut)
def update_food_log(
    log_id: str,
    food_log_update: FoodLogUpdate,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    db_food_log = (
        db.query(FoodLog)
        .filter(FoodLog.id == log_id, FoodLog.user_id == current_user.id)
        .first()
    )
    if not db_food_log:
        raise HTTPException(status_code=404, detail="Food log not found")

    for key, value in food_log_update.dict(exclude_unset=True).items():
        setattr(db_food_log, key, value)

    db.commit()
    db.refresh(db_food_log)
    upsert_daily_summary(db, db_food_log.logged_at, current_user.id)
    return db_food_log


@router.delete("/api/v1/logs/{log_id}", status_code=204)
def delete_food_log(
    log_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    db_food_log = (
        db.query(FoodLog)
        .filter(FoodLog.id == log_id, FoodLog.user_id == current_user.id)
        .first()
    )
    if not db_food_log:
        raise HTTPException(status_code=404, detail="Food log not found")

    day = db_food_log.logged_at
    db.delete(db_food_log)
    db.commit()
    upsert_daily_summary(db, day, current_user.id)
    return {"message": "Food log deleted successfully"}
