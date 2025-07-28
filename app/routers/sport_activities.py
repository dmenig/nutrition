from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
from uuid import UUID

from app import schemas
from app.db.models import SportActivity
from app.db.database import get_db
from app.core.auth import get_current_user

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
    db_sport_activity = SportActivity(
        **sport_activity.model_dump(), owner_id=current_user.id
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

    sport_activities = (
        db.query(SportActivity)
        .filter(
            SportActivity.owner_id == current_user.id, SportActivity.date == parsed_date
        )
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
            SportActivity.id == activity_id, SportActivity.owner_id == current_user.id
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
