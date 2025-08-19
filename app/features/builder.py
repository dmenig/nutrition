from __future__ import annotations

import os
import pathlib
from functools import lru_cache
from typing import Dict, List

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import FoodLog, SportActivity, User, WeightLog


@lru_cache(maxsize=1)
def load_variables_lookup() -> Dict[str, Dict[str, float]]:
    """Load variables.csv and return per-100g nutrient map by food name (original CSV name).

    Keys include: 'Calories / 100g', 'Carbs', 'Sugar', 'Sel', 'Alcool', 'Water'.
    """
    candidates = [
        os.path.join("/app", "data", "variables.csv"),
        os.path.join(pathlib.Path(__file__).resolve().parents[2], "data", "variables.csv"),
        os.path.join(os.getcwd(), "data", "variables.csv"),
    ]
    csv_path = next((p for p in candidates if os.path.exists(p)), None)
    if not csv_path:
        return {}
    df = pd.read_csv(csv_path)
    lookup: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        name = str(row.get("Nom", "")).strip()
        if not name:
            continue
        lookup[name] = {
            "Calories / 100g": float(row.get("Calories / 100g", 0.0) or 0.0),
            "Carbs": float(row.get("Carbs", 0.0) or 0.0),
            "Sugar": float(row.get("Sugar", 0.0) or 0.0),
            "Sel": float(row.get("Sel", 0.0) or 0.0),
            "Alcool": float(row.get("Alcool", 0.0) or 0.0),
            "Water": float(row.get("Water", 0.0) or 0.0),
        }
    return lookup


def compute_nutrients_by_date(db: Session, dates: List, user_id) -> Dict:
    """Compute per-date nutrient sums using FoodLog.quantity join with variables.csv.

    Returns a mapping: date -> {calories, carbs, sugar, sel, alcool, water}
    """
    lookup = load_variables_lookup()
    base = {
        "calories": 0.0,
        "carbs": 0.0,
        "sugar": 0.0,
        "sel": 0.0,
        "alcool": 0.0,
        "water": 0.0,
    }
    by_date = {d: dict(base) for d in dates}
    if not lookup or not dates:
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
        acc["calories"] += per100["Calories / 100g"] * scale
        acc["carbs"] += per100["Carbs"] * scale
        acc["sugar"] += per100["Sugar"] * scale
        acc["sel"] += per100["Sel"] * scale
        acc["alcool"] += per100["Alcool"] * scale
        acc["water"] += per100["Water"] * scale
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
        return pd.DataFrame(recs)
    finally:
        db.close()


