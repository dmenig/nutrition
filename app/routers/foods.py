from typing import List

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Food
from app.schemas import FoodOut

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
