import os
import json
import pickle
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import FoodLog, SportActivity
from train_model import train_and_save_model
from app.core.config import settings  # Assuming settings contains DATABASE_URL

MODEL_DIR = "/app/models"  # Persistent location for deployed container


def retrain_model_job():
    # Ensure the model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Fetch historical food and sport data
        food_logs = db.query(FoodLog).all()
        sport_activities = db.query(SportActivity).all()

        # Convert SQLAlchemy objects to Pandas DataFrames
        food_data_df = pd.DataFrame([log.__dict__ for log in food_logs])
        sport_data_df = pd.DataFrame(
            [activity.__dict__ for activity in sport_activities]
        )

        # Drop SQLAlchemy internal state if present
        if "_sa_instance_state" in food_data_df.columns:
            food_data_df = food_data_df.drop(columns=["_sa_instance_state"])
        if "_sa_instance_state" in sport_data_df.columns:
            sport_data_df = sport_data_df.drop(columns=["_sa_instance_state"])

        # Define save paths (align with app expectations)
        model_path = os.path.join(MODEL_DIR, "recurrent_model.pth")
        params_path = os.path.join(MODEL_DIR, "best_params.json")

        # Run the training logic
        model, best_params = train_and_save_model(
            food_data_df,
            sport_data_df,
            model_save_path=model_path,
            params_save_path=params_path,
        )

        print("Model retraining completed successfully.")
    except Exception as e:
        print(f"Error during model retraining: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    retrain_model_job()
