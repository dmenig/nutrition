from typing import List
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import CustomFood, User
from app.schemas import CustomFoodCreate, CustomFoodOut, CustomFoodUpdate
from app.core.auth import get_current_user
from app.db.database import get_db

router = APIRouter()


@router.post(
    "/api/v1/custom-foods",
    response_model=CustomFoodOut,
    status_code=status.HTTP_201_CREATED,
)
def create_custom_food(
    food: CustomFoodCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db_food = CustomFood(
        id=uuid4(),
        name=food.name,
        calories=food.calories,
        protein=food.protein,
        carbohydrates=food.carbohydrates,
        fat=food.fat,
        owner_id=current_user.id,
    )
    db.add(db_food)
    db.commit()
    db.refresh(db_food)
    return db_food


@router.get("/api/v1/custom-foods", response_model=List[CustomFoodOut])
def read_custom_foods(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.query(CustomFood).filter(CustomFood.owner_id == current_user.id).all()


@router.delete("/api/v1/custom-foods/{food_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_custom_food(
    food_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db_food = (
        db.query(CustomFood)
        .filter(CustomFood.id == food_id, CustomFood.owner_id == current_user.id)
        .first()
    )
    if not db_food:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Custom food not found"
        )
    db.delete(db_food)
    db.commit()
    return {"detail": "Custom food deleted"}
