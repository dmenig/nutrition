import uuid
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.models import DailySummary, FoodLog, SportActivity


def upsert_daily_summary(db: Session, target_date: datetime, user_id: uuid.UUID | None = None) -> None:
    day_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    food_q = db.query(
        func.coalesce(func.sum(FoodLog.calories), 0.0),
        func.coalesce(func.sum(FoodLog.protein), 0.0),
        func.coalesce(func.sum(FoodLog.carbs), 0.0),
        func.coalesce(func.sum(FoodLog.fat), 0.0),
    ).filter(FoodLog.logged_at >= day_start, FoodLog.logged_at < day_end)
    if user_id:
        food_q = food_q.filter(FoodLog.user_id == user_id)
    calories, protein, carbs, fat = food_q.first()

    sport_q = db.query(func.coalesce(func.sum(SportActivity.calories_expended), 0.0)).filter(
        SportActivity.logged_at >= day_start, SportActivity.logged_at < day_end
    )
    if user_id:
        sport_q = sport_q.filter(SportActivity.user_id == user_id)
    sport_total = sport_q.scalar() or 0.0

    summary = (
        db.query(DailySummary)
        .filter(DailySummary.date == day_start.date(), DailySummary.user_id == user_id)
        .first()
    )
    if summary is None:
        summary = DailySummary(
            user_id=user_id,
            date=day_start.date(),
            calories_total=calories,
            protein_g_total=protein,
            carbs_g_total=carbs,
            fat_g_total=fat,
            sport_calories_total=sport_total,
        )
        db.add(summary)
    else:
        summary.calories_total = calories
        summary.protein_g_total = protein
        summary.carbs_g_total = carbs
        summary.fat_g_total = fat
        summary.sport_calories_total = sport_total
    db.commit()


