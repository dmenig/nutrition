from __future__ import annotations

import os
import pathlib
from functools import lru_cache
from typing import Dict, List

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import FoodLog, SportActivity, User, WeightLog, Food


@lru_cache(maxsize=1)
def load_variables_lookup() -> Dict[str, Dict[str, float]]:
    """Deprecated: CSV lookups are not used in server runtime. Returns empty dict."""
    return {}


def compute_nutrients_by_date(db: Session, dates: List, user_id) -> Dict:
    """Compute per-date nutrient sums using only the database.

    - calories, carbs: sum directly from FoodLog columns (already absolute values)
    - sugar, sel, alcool, water: join to Food by name and scale by quantity/100
    """
    base = {
        "calories": 0.0,
        "carbs": 0.0,
        "sugar": 0.0,
        "sel": 0.0,
        "alcool": 0.0,
        "water": 0.0,
    }
    by_date = {d: dict(base) for d in dates}
    if not dates:
        return by_date

    dcol = func.coalesce(FoodLog.logged_date, func.date(FoodLog.logged_at))
    q = (
        db.query(
            dcol.label("d"),
            func.coalesce(func.sum(FoodLog.calories), 0.0).label("calories"),
            func.coalesce(func.sum(FoodLog.carbs), 0.0).label("carbs"),
            # Scale per-100g nutrient columns by quantity (grams)
            func.coalesce(func.sum((Food.sugar * (FoodLog.quantity / 100.0))), 0.0).label("sugar"),
            func.coalesce(func.sum((Food.sel * (FoodLog.quantity / 100.0))), 0.0).label("sel"),
            func.coalesce(func.sum((Food.alcool * (FoodLog.quantity / 100.0))), 0.0).label("alcool"),
            func.coalesce(func.sum((Food.water * (FoodLog.quantity / 100.0))), 0.0).label("water"),
        )
        .outerjoin(Food, Food.name == FoodLog.food_name)
    )
    if user_id:
        q = q.filter(FoodLog.user_id == user_id)
    q = q.filter(dcol.in_(dates))
    q = q.group_by("d").all()
    for r in q:
        acc = by_date.get(r.d)
        if acc is None:
            continue
        acc["calories"] = float(getattr(r, "calories", 0.0) or 0.0)
        acc["carbs"] = float(getattr(r, "carbs", 0.0) or 0.0)
        acc["sugar"] = float(getattr(r, "sugar", 0.0) or 0.0)
        acc["sel"] = float(getattr(r, "sel", 0.0) or 0.0)
        acc["alcool"] = float(getattr(r, "alcool", 0.0) or 0.0)
        acc["water"] = float(getattr(r, "water", 0.0) or 0.0)
    return by_date
    q = db.query(
        FoodLog.food_name,
        FoodLog.quantity,
        func.coalesce(FoodLog.logged_date, func.date(FoodLog.logged_at)).label("d"),
    )
    if user_id:
        q = q.filter(FoodLog.user_id == user_id)
    q = q.filter(func.coalesce(FoodLog.logged_date, func.date(FoodLog.logged_at)).in_(dates))
    for name, qty_g, d in q.all():
        per100 = lookup.get(str(name))
        if not per100:
            continue
        scale = float(qty_g or 0.0) / 100.0
        acc = by_date.get(d)
        if acc is None:
            continue
        acc["calories"] += float(per100.get("Calories / 100g", 0.0) or 0.0) * scale
        acc["carbs"] += float(per100.get("Carbs", 0.0) or 0.0) * scale
        acc["sugar"] += float(per100.get("Sugar", 0.0) or 0.0) * scale
        acc["sel"] += float(per100.get("Sel", 0.0) or 0.0) * scale
        acc["alcool"] += float(per100.get("Alcool", 0.0) or 0.0) * scale
        acc["water"] += float(per100.get("Water", 0.0) or 0.0) * scale
    return by_date


def build_prediction_features_from_db() -> pd.DataFrame:
    """Build the full features DataFrame from DB, matching the CSV pipeline semantics.

    - Scope to dummy@example.com when present
    - Exclude today's date
    - Derive nutrients from variables.csv join
    - Aggregate sport calories per day
    - Use daily average observed weights (pds)
    """
    from app.db.database import SessionLocal  # lazy import

    db = SessionLocal()
    try:
        dummy = db.query(User).filter(User.email == "dummy@example.com").first()
        dummy_id = getattr(dummy, "id", None)

        dfl = func.coalesce(FoodLog.logged_date, func.date(FoodLog.logged_at))
        food_dates_q = db.query(dfl.label("date"))
        if dummy_id:
            food_dates_q = food_dates_q.filter(FoodLog.user_id == dummy_id)
        food_dates = [r.date for r in food_dates_q.group_by(dfl).all()]

        dsa = func.coalesce(SportActivity.logged_date, func.date(SportActivity.logged_at))
        sport_dates_q = db.query(dsa.label("date"))
        if dummy_id:
            sport_dates_q = sport_dates_q.filter(SportActivity.user_id == dummy_id)
        sport_dates = [r.date for r in sport_dates_q.group_by(dsa).all()]

        dates = sorted(set(food_dates) | set(sport_dates))
        today = pd.Timestamp.utcnow().date()
        dates = [d for d in dates if d < today]

        # Observed weight per day
        dwt = func.coalesce(WeightLog.logged_date, func.date(WeightLog.logged_at))
        wq = db.query(dwt.label("date"), func.avg(WeightLog.weight_kg).label("pds"))
        if dummy_id:
            wq = wq.filter(WeightLog.user_id == dummy_id)
        weight_rows = wq.group_by(dwt).all()
        weight_by_date = {r.date: float(getattr(r, "pds", 0.0) or 0.0) for r in weight_rows}

        # Nutrients from variables.csv join
        nutr_by_date = compute_nutrients_by_date(db, dates, dummy_id)

        # Sport per day
        sport_by_date = {}
        srows = db.query(
            dsa.label("date"), func.coalesce(func.sum(SportActivity.calories_expended), 0.0).label("sport")
        )
        if dummy_id:
            srows = srows.filter(SportActivity.user_id == dummy_id)
        srows = srows.group_by(dsa).all()
        for r in srows:
            sport_by_date[r.date] = float(getattr(r, "sport", 0.0) or 0.0)

        # Assemble records
        recs: List[dict] = []
        for d in dates:
            n = nutr_by_date.get(d, {})
            recs.append(
                {
                    "date": d,
                    "calories": float(n.get("calories", 0.0)),
                    "carbs": float(n.get("carbs", 0.0)),
                    "sugar": float(n.get("sugar", 0.0)),
                    "sel": float(n.get("sel", 0.0)),
                    "alcool": float(n.get("alcool", 0.0)),
                    "water": float(n.get("water", 0.0)),
                    "sport": float(sport_by_date.get(d, 0.0)),
                    "pds": float(weight_by_date.get(d, 0.0)),
                }
            )
        df = pd.DataFrame(recs)
        # NA coercion: ensure no NaN/NA leak into inference pipeline
        for col in ["calories","carbs","sugar","sel","alcool","water","sport","pds"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df
    finally:
        db.close()


