from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.database import engine, Base, get_db
from app.routers import (
    auth,
    custom_foods,
    sport_activities,
    food_logs,
    admin,
)
from app.schemas import (
    WeightPlotResponse,
    MetabolismPlotResponse,
    EnergyBalancePlotResponse,
    PlotPoint,
)
import pandas as pd
import json
import os
from typing import Any
from fastapi import BackgroundTasks
import torch
from train_model import FinalModel  # Import the model class
import json  # Added import for json

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(auth.router, prefix="/api/v1", tags=["users"])
app.include_router(custom_foods.router, prefix="", tags=["custom-foods"])
app.include_router(sport_activities.router, prefix="", tags=["sports"])
app.include_router(food_logs.router, prefix="", tags=["food-logs"])


class PredictionService:
    model: Any = None
    model_path: str = "/app/models/recurrent_model.pth"
    params_path: str = "/app/models/best_params.json"
    normalization_stats: dict = {}  # Initialize normalization_stats

    def __init__(self):
        self.load_model()

    def load_model(self):
        # Load normalization stats and initial weight guess from best_params.json
        try:
            with open(self.params_path, "r") as f:
                params = json.load(f)
                self.normalization_stats = params["normalization"]  # Store in self
                # Assuming initial_weight_guess is part of normalization_stats or derived
                # For now, we'll use a dummy value or derive it if needed.
                # A more robust solution would save this with the model params.
                # If 'pds' mean is available, we can use it as a proxy for initial_weight_guess
                initial_weight_guess = (
                    self.normalization_stats["pds"]["mean"]
                    if "pds" in self.normalization_stats
                    else 70.0
                )  # Default if not found
        except FileNotFoundError:
            print(
                f"Warning: {self.params_path} not found. Using default initial_weight_guess."
            )
            initial_weight_guess = 70.0  # Fallback default
        except json.JSONDecodeError:
            print(
                f"Warning: Could not decode JSON from {self.params_path}. Using default initial_weight_guess."
            )
            initial_weight_guess = 70.0  # Fallback default

        # Instantiate the model with appropriate parameters
        # Assuming nutrition_input_size can be inferred or is a fixed value
        # For now, let's use a placeholder. This needs to match the training model.
        # From train_model.py, nutrition_input_size is nutrition_data.shape[2]
        # which is len(nutrition_cols) = 6.
        nutrition_input_size = 6
        self.model = FinalModel(nutrition_input_size, initial_weight_guess)

        # Load the state dictionary
        # Use map_location=torch.device('cpu') to load CPU-only if trained on GPU
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=torch.device("cpu"))
        )
        self.model.eval()  # Set model to evaluation mode
        print(f"Model loaded successfully from {self.model_path}")

    def predict(self, data: pd.DataFrame):
        if self.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction model not loaded.",
            )
        # Assuming the input data needs to be converted to a tensor and normalized
        # This part needs to be aligned with how the model expects its input during prediction
        # For now, a placeholder for prediction.
        # The actual prediction logic will be more complex, involving data normalization
        # and tensor conversion similar to train_model.py
        # return self.model.predict(data).tolist() # This line is incorrect for a PyTorch model

        # Placeholder for actual prediction logic
        # Convert pandas DataFrame to torch tensor
        # Ensure data has the same columns and order as during training
        # This is a simplified example and needs to be expanded based on actual model input

        # Example: Assuming 'data' DataFrame contains the nutrition_cols
        nutrition_cols = ["calories", "carbs", "sugar", "sel", "alcool", "water"]

        # Ensure data has the correct columns, fill missing with 0, convert to numeric
        for col in nutrition_cols:
            if col not in data.columns:
                data[col] = 0.0
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

        # Normalize data using the loaded normalization_stats
        normalized_data = data[nutrition_cols].copy()
        for col in nutrition_cols:
            if (
                col in self.normalization_stats
                and self.normalization_stats[col]["std"] != 0
            ):
                normalized_data[col] = (
                    normalized_data[col] - self.normalization_stats[col]["mean"]
                ) / self.normalization_stats[col]["std"]
            else:
                # Handle cases where std is 0 or stats are missing (e.g., for constant features)
                normalized_data[col] = (
                    normalized_data[col] - self.normalization_stats[col]["mean"]
                )  # Just center it if std is 0

        nutrition_tensor = torch.tensor(
            normalized_data.values, dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            # The FinalModel's forward method returns base_metabolisms
            # We need to reconstruct the trajectory to get predicted weights
            base_metabolisms = self.model(nutrition_tensor)

            # To reconstruct trajectory, we also need observed_weights, sport_data, initial_adj_weight
            # These would typically come from the context of the prediction request or be defaults.
            # For a simple prediction endpoint, we might only predict metabolism or a single weight.
            # If the goal is to predict a trajectory, more input is needed.

            # For now, let's just return the base metabolisms as a placeholder for prediction output
            # This needs to be refined based on what the /predict/latest endpoint is supposed to return.
            return base_metabolisms.squeeze().tolist()


@app.get("/api/v1/health")
def health_check():
    return {"status": "ok"}


prediction_service = PredictionService()


@app.on_event("startup")
async def startup_event():
    prediction_service.load_model()


@app.get("/api/v1/predict/latest", tags=["prediction"])
async def get_latest_prediction():
    dummy_data = pd.DataFrame(
        [[1.0, 2.0, 3.0]], columns=["feature1", "feature2", "feature3"]
    )
    prediction = prediction_service.predict(dummy_data)
    return {"prediction": prediction}


@app.post("/api/v1/predict/reload-model", tags=["prediction"])
async def reload_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(prediction_service.load_model)
    return {"message": "Model reload initiated in the background."}


app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])


def get_plot_data():
    try:
        df = pd.read_csv("data/final_results.csv")
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Results file not found. Please run train_model.py first.",
        )

    df["time_index"] = range(len(df))

    # De-normalize calories and sport for plotting
    try:
        with open("best_params.json", "r") as f:
            params = json.load(f)
            normalization_stats = params["normalization"]
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Normalization parameters not found. Please run train_model.py first.",
        )

    df["calories_unnorm"] = (
        df["calories"] * normalization_stats["calories"]["std"]
        + normalization_stats["calories"]["mean"]
    )
    df["sport_unnorm"] = (
        df["sport"] * normalization_stats["sport"]["std"]
        + normalization_stats["sport"]["mean"]
    )
    df["C_exp_t"] = df["M_base"] + df["sport_unnorm"]
    return df


@app.get("/api/v1/plots/weight", response_model=WeightPlotResponse, tags=["plots"])
def get_weight_plot_data():
    df = get_plot_data()
    return WeightPlotResponse(
        W_obs=[
            {"time_index": row["time_index"], "value": row["W_obs"]}
            for index, row in df.iterrows()
        ],
        W_adj_pred=[
            {"time_index": row["time_index"], "value": row["W_adj_pred"]}
            for index, row in df.iterrows()
        ],
    )


@app.get(
    "/api/v1/plots/metabolism", response_model=MetabolismPlotResponse, tags=["plots"]
)
def get_metabolism_plot_data():
    df = get_plot_data()
    return MetabolismPlotResponse(
        M_base=[
            {"time_index": row["time_index"], "value": row["M_base"]}
            for index, row in df.iterrows()
        ]
    )


@app.get(
    "/api/v1/plots/energy-balance",
    response_model=EnergyBalancePlotResponse,
    tags=["plots"],
)
def get_energy_balance_plot_data():
    df = get_plot_data()
    return EnergyBalancePlotResponse(
        calories_unnorm=[
            {"time_index": row["time_index"], "value": row["calories_unnorm"]}
            for index, row in df.iterrows()
        ],
        C_exp_t=[
            {"time_index": row["time_index"], "value": row["C_exp_t"]}
            for index, row in df.iterrows()
        ],
    )
