from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timezone, timedelta
from uuid import UUID
from sqlalchemy import func

from app import schemas
from app.db.models import SportActivity, User
from app.db.database import get_db
from app.core.auth import get_current_user
from sport_formulas import MET_VALUES, evaluate_sport_formula

router = APIRouter()


@router.post(
    "/api/v1/sports",
    response_model=schemas.SportActivityOut,
    status_code=status.HTTP_201_CREATED,
)
def create_sport_activity(
    sport_activity: schemas.SportActivityCreate,
    db: Session = Depends(get_db),
    current_user: schemas.UserOut = Depends(get_current_user),
):
    user_weight = (
        db.query(User.current_weight_kg).filter(User.id == current_user.id).scalar()
    )
    if user_weight is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User weight not found. Please update your profile with your current weight.",
        )

    calories_expended = evaluate_sport_formula(
        f"{sport_activity.activity_name.upper()}_CALORIES("
        f"duration_minutes={sport_activity.duration_minutes},"
        f"weight_kg={user_weight},"
        f"distance_meters={sport_activity.distance_m or 0},"
        f"additional_weight_kg={sport_activity.carried_weight_kg or 0})"
    )

    db_sport_activity = SportActivity(
        activity_name=sport_activity.activity_name,
        logged_at=sport_activity.logged_at,
        duration_minutes=sport_activity.duration_minutes,
        carried_weight_kg=sport_activity.carried_weight_kg,
        distance_m=sport_activity.distance_m,
        calories_expended=calories_expended,
        user_id=current_user.id,
    )
    db.add(db_sport_activity)
    db.commit()
    db.refresh(db_sport_activity)
    return db_sport_activity


@router.get("/api/v1/sports", response_model=List[schemas.SportActivityOut])
def get_sport_activities_by_date(
    date: str,
    db: Session = Depends(get_db),
    current_user: schemas.UserOut = Depends(get_current_user),
):
    try:
        parsed_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Please use YYYY-MM-DD.",
        )

    # Use range scan to allow index usage
    day_start = datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    sport_activities = (
        db.query(SportActivity)
        .filter(
            SportActivity.user_id == current_user.id,
            SportActivity.logged_at >= day_start,
            SportActivity.logged_at < day_end,
        )
        .order_by(SportActivity.logged_at.asc())
        .all()
    )
    return sport_activities


@router.delete("/api/v1/sports/{activity_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_sport_activity(
    activity_id: UUID,
    db: Session = Depends(get_db),
    current_user: schemas.UserOut = Depends(get_current_user),
):
    db_sport_activity = (
        db.query(SportActivity)
        .filter(
            SportActivity.id == activity_id, SportActivity.user_id == current_user.id
        )
        .first()
    )

    if not db_sport_activity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Sport activity not found."
        )

    db.delete(db_sport_activity)
    db.commit()
    return


@router.get("/api/v1/sports/names", response_model=List[str])
def get_sport_names():
    return list(MET_VALUES.keys())


@router.get("/api/v1/sports/public", response_model=List[schemas.SportActivityOut])
def get_sport_activities_by_date_public(
    date: str,
    db: Session = Depends(get_db),
):
    try:
        parsed_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Please use YYYY-MM-DD.",
        )

    dummy_user = db.query(User).filter(User.email == "dummy@example.com").first()

    day_start = datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    query = db.query(SportActivity).filter(
        SportActivity.logged_at >= day_start,
        SportActivity.logged_at < day_end,
    )
    if dummy_user:
        query = query.filter(SportActivity.user_id == dummy_user.id)

    sport_activities = query.order_by(SportActivity.logged_at.asc()).all()
    return sport_activities
