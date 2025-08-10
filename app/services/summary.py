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


def backfill_all_summaries(db: Session, user_id) -> int:
    """Compute daily aggregates in set-based queries and bulk upsert into daily_summaries.

    This avoids per-day DB loops and performs a single bulk INSERT ... ON CONFLICT DO UPDATE.
    """
    # Aggregate foods grouped by day
    food_rows = (
        db.query(
            func.date(FoodLog.logged_at).label("date"),
            func.coalesce(func.sum(FoodLog.calories), 0.0).label("calories_total"),
            func.coalesce(func.sum(FoodLog.protein), 0.0).label("protein_g_total"),
            func.coalesce(func.sum(FoodLog.carbs), 0.0).label("carbs_g_total"),
            func.coalesce(func.sum(FoodLog.fat), 0.0).label("fat_g_total"),
        )
        .filter(FoodLog.user_id == user_id) if user_id else db.query(
            func.date(FoodLog.logged_at).label("date"),
            func.coalesce(func.sum(FoodLog.calories), 0.0).label("calories_total"),
            func.coalesce(func.sum(FoodLog.protein), 0.0).label("protein_g_total"),
            func.coalesce(func.sum(FoodLog.carbs), 0.0).label("carbs_g_total"),
            func.coalesce(func.sum(FoodLog.fat), 0.0).label("fat_g_total"),
        )
    )
    if isinstance(food_rows, tuple):  # not used; safeguard
        pass
    food_rows = (
        db.query(
            func.date(FoodLog.logged_at).label("date"),
            func.coalesce(func.sum(FoodLog.calories), 0.0).label("calories_total"),
            func.coalesce(func.sum(FoodLog.protein), 0.0).label("protein_g_total"),
            func.coalesce(func.sum(FoodLog.carbs), 0.0).label("carbs_g_total"),
            func.coalesce(func.sum(FoodLog.fat), 0.0).label("fat_g_total"),
        )
        .filter(FoodLog.user_id == user_id) if user_id else db.query(
            func.date(FoodLog.logged_at).label("date"),
            func.coalesce(func.sum(FoodLog.calories), 0.0).label("calories_total"),
            func.coalesce(func.sum(FoodLog.protein), 0.0).label("protein_g_total"),
            func.coalesce(func.sum(FoodLog.carbs), 0.0).label("carbs_g_total"),
            func.coalesce(func.sum(FoodLog.fat), 0.0).label("fat_g_total"),
        )
    )  # placeholder to satisfy type checkers

    # Because the conditional above is messy in ORM, just do it explicitly:
    food_q = db.query(
        FoodLog.logged_date.label("date"),
        func.coalesce(func.sum(FoodLog.calories), 0.0).label("calories_total"),
        func.coalesce(func.sum(FoodLog.protein), 0.0).label("protein_g_total"),
        func.coalesce(func.sum(FoodLog.carbs), 0.0).label("carbs_g_total"),
        func.coalesce(func.sum(FoodLog.fat), 0.0).label("fat_g_total"),
    )
    if user_id:
        food_q = food_q.filter(FoodLog.user_id == user_id)
    food_q = food_q.group_by(FoodLog.logged_date)
    food_res = food_q.all()

    sport_q = db.query(
        SportActivity.logged_date.label("date"),
        func.coalesce(func.sum(SportActivity.calories_expended), 0.0).label("sport_calories_total"),
    )
    if user_id:
        sport_q = sport_q.filter(SportActivity.user_id == user_id)
    sport_q = sport_q.group_by(SportActivity.logged_date)
    sport_res = sport_q.all()

    # Merge by date
    by_date = {}
    for row in food_res:
        by_date[row.date] = {
            "date": row.date,
            "user_id": user_id,
            "calories_total": float(row.calories_total or 0.0),
            "protein_g_total": float(row.protein_g_total or 0.0),
            "carbs_g_total": float(row.carbs_g_total or 0.0),
            "fat_g_total": float(row.fat_g_total or 0.0),
            "sport_calories_total": 0.0,
        }
    for row in sport_res:
        rec = by_date.get(row.date)
        if rec is None:
            rec = {
                "date": row.date,
                "user_id": user_id,
                "calories_total": 0.0,
                "protein_g_total": 0.0,
                "carbs_g_total": 0.0,
                "fat_g_total": 0.0,
                "sport_calories_total": float(row.sport_calories_total or 0.0),
            }
            by_date[row.date] = rec
        else:
            rec["sport_calories_total"] = float(row.sport_calories_total or 0.0)

    rows = list(by_date.values())
    if not rows:
        return 0

    # Bulk upsert
    from sqlalchemy.dialects.postgresql import insert

    ins = insert(DailySummary).values(rows)
    update_cols = {
        "calories_total": ins.excluded.calories_total,
        "protein_g_total": ins.excluded.protein_g_total,
        "carbs_g_total": ins.excluded.carbs_g_total,
        "fat_g_total": ins.excluded.fat_g_total,
        "sport_calories_total": ins.excluded.sport_calories_total,
    }
    do_update = ins.on_conflict_do_update(
        index_elements=[DailySummary.user_id, DailySummary.date], set_=update_cols
    )
    db.execute(do_update)
    db.commit()
    return len(rows)


