import os
from sqlalchemy import create_engine, text
from app.core.config import settings


def verify_population():
    engine = create_engine(settings.DATABASE_URL)
    with engine.connect() as connection:
        food_count = connection.execute(text("SELECT COUNT(*) FROM foods")).scalar_one()
        food_log_count = connection.execute(
            text("SELECT COUNT(*) FROM food_logs")
        ).scalar_one()
        sport_count = connection.execute(
            text("SELECT COUNT(*) FROM sport_activities")
        ).scalar_one()
        print(f"Rows in foods: {food_count}")
        print(f"Rows in food_logs: {food_log_count}")
        print(f"Rows in sport_activities: {sport_count}")


if __name__ == "__main__":
    verify_population()
