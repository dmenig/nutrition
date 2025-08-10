import os
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.database import SessionLocal
from app.db.models import FoodLog, SportActivity, User
from app.services.summary import upsert_daily_summary


def backfill() -> int:
    db: Session = SessionLocal()
    try:
        dummy_user = db.query(User).filter(User.email == "dummy@example.com").first()
        user_id = dummy_user.id if dummy_user else None

        food_dates = db.query(func.date(FoodLog.logged_at)).distinct().all()
        sport_dates = db.query(func.date(SportActivity.logged_at)).distinct().all()
        dates = {d[0] for d in food_dates} | {d[0] for d in sport_dates}

        count = 0
        for d in sorted(dates):
            target = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
            upsert_daily_summary(db, target, user_id)
            count += 1
        return count
    finally:
        db.close()


if __name__ == "__main__":
    updated = backfill()
    print(f"Backfilled {updated} day(s)")


