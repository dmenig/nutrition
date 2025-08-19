from typing import List

from fastapi import APIRouter, Depends, Query, status, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Food
from app.schemas import FoodOut, FoodCreate

router = APIRouter()


@router.get("/api/v1/foods/search", response_model=List[FoodOut])
def search_foods(query: str = Query(..., min_length=1), db: Session = Depends(get_db)):
    foods = db.query(Food).filter(Food.name.ilike(f"%{query}%")).all()
    return foods


@router.get("/api/v1/foods/autocomplete", response_model=List[FoodOut])
def autocomplete_foods(
    q: str = Query(..., min_length=1), db: Session = Depends(get_db)
):
    foods = (
        db.query(Food)
        .filter(Food.name.ilike(f"{q}%"))
        .order_by(Food.name)
        .limit(10)
        .all()
    )
    return foods


@router.post("/api/v1/foods", response_model=FoodOut, status_code=status.HTTP_201_CREATED)
def create_food(food: FoodCreate, db: Session = Depends(get_db)):
    # Enforce unique by name
    existing = db.query(Food).filter(Food.name.ilike(food.name)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Food with this name already exists")
    db_food = Food(
        name=food.name,
        calories=food.calories,
        protein=food.protein,
        carbs=food.carbs,
        fat=food.fat,
    )
    db.add(db_food)
    db.commit()
    db.refresh(db_food)
    return db_food


@router.delete("/api/v1/foods/{food_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_food(food_id: str, db: Session = Depends(get_db)):
    item = db.query(Food).filter(Food.id == food_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Food not found")
    db.delete(item)
    db.commit()
    return {"detail": "Deleted"}
