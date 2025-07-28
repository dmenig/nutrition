from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, date
from sqlalchemy import func  # Import func for aggregation

from app.db.models import FoodLog
from app.schemas import (
    FoodLogCreate,
    FoodLogUpdate,
    FoodLogOut,
    DailyFoodLogSummary,
)  # Import DailyFoodLogSummary
from app.core.auth import get_current_user
from app.db.database import get_db

router = APIRouter()


@router.get("/api/v1/logs/summary", response_model=DailyFoodLogSummary)
def get_daily_food_log_summary(
    date: date = Query(..., description="Date in YYYY-MM-DD format"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    summary = (
        db.query(
            func.sum(FoodLog.calories).label("calories"),
            func.sum(FoodLog.protein).label("protein_g"),
            func.sum(FoodLog.carbs).label("carbs_g"),
            func.sum(FoodLog.fat).label("fat_g"),
        )
        .filter(
            FoodLog.user_id == current_user["id"], func.date(FoodLog.logged_at) == date
        )
        .group_by(func.date(FoodLog.logged_at))
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
    current_user: dict = Depends(get_current_user),
):
    db_food_log = FoodLog(**food_log.dict(), user_id=current_user["id"])
    db.add(db_food_log)
    db.commit()
    db.refresh(db_food_log)
    return db_food_log


@router.get("/api/v1/logs", response_model=List[FoodLogOut])
def get_food_logs_for_date(
    date: date = Query(..., description="Date in YYYY-MM-DD format"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    food_logs = (
        db.query(FoodLog)
        .filter(FoodLog.user_id == current_user["id"], FoodLog.logged_at == date)
        .all()
    )
    return food_logs


@router.put("/api/v1/logs/{log_id}", response_model=FoodLogOut)
def update_food_log(
    log_id: str,
    food_log_update: FoodLogUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    db_food_log = (
        db.query(FoodLog)
        .filter(FoodLog.id == log_id, FoodLog.user_id == current_user["id"])
        .first()
    )
    if not db_food_log:
        raise HTTPException(status_code=404, detail="Food log not found")

    for key, value in food_log_update.dict(exclude_unset=True).items():
        setattr(db_food_log, key, value)

    db.commit()
    db.refresh(db_food_log)
    return db_food_log


@router.delete("/api/v1/logs/{log_id}", status_code=204)
def delete_food_log(
    log_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    db_food_log = (
        db.query(FoodLog)
        .filter(FoodLog.id == log_id, FoodLog.user_id == current_user["id"])
        .first()
    )
    if not db_food_log:
        raise HTTPException(status_code=404, detail="Food log not found")

    db.delete(db_food_log)
    db.commit()
    return {"message": "Food log deleted successfully"}
