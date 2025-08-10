import os
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.database import SessionLocal
from app.db.models import FoodLog, SportActivity, User
from app.services.summary import upsert_daily_summary, backfill_all_summaries


def backfill() -> int:
    db: Session = SessionLocal()
    try:
        dummy_user = db.query(User).filter(User.email == "dummy@example.com").first()
        user_id = dummy_user.id if dummy_user else None

        return backfill_all_summaries(db, user_id)
    finally:
        db.close()


if __name__ == "__main__":
    updated = backfill()
    print(f"Backfilled {updated} day(s)")


