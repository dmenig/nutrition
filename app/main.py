from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from app.db.database import engine, Base, get_db
from app.routers import (
    auth,
    custom_foods,
    sport_activities,
    food_logs,
    admin,
    foods,
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
from train_model import FinalModel, reconstruct_trajectory  # Import the model and trajectory util
import json  # Added import for json

Base.metadata.create_all(bind=engine)

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(auth.router, prefix="/api/v1", tags=["users"])
app.include_router(custom_foods.router, prefix="", tags=["custom-foods"])
app.include_router(sport_activities.router, prefix="", tags=["sports"])
app.include_router(food_logs.router, prefix="", tags=["food-logs"])
app.include_router(foods.router, prefix="", tags=["foods"])


class PredictionService:
    model: Any = None
    model_path: str = "models/recurrent_model.pth"
    params_path: str = "models/best_params.json"
    normalization_stats: dict = {}  # Initialize normalization_stats

    def __init__(self):
        self.load_model()

    def _resolve_first_existing(self, candidates):
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def load_model(self):
        # Load normalization stats and initial weight guess from best_params.json
        try:
            params_path = self._resolve_first_existing(
                [self.params_path, "/app/best_params.json", "/app/models/best_params.json"]
            )
            with open(params_path, "r") as f:
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

        # Load weights from known locations
        model_path = self._resolve_first_existing(
            [
                self.model_path,
                "/app/recurrent_model.pth",
                "/app/models/recurrent_model.pth",
                "models/model.pkl",
                "/app/models/model.pkl",
            ]
        )
        loaded = torch.load(model_path, map_location=torch.device("cpu"))
        try:
            # Assume state dict
            self.model.load_state_dict(loaded)
        except Exception:
            # Fallback: maybe a dict wrapper
            if isinstance(loaded, dict) and "state_dict" in loaded:
                self.model.load_state_dict(loaded["state_dict"])
            else:
                # As a last resort, try loading a full pickled nn.Module
                try:
                    self.model = loaded
                except Exception as e:
                    raise RuntimeError(f"Failed to load model weights from {model_path}: {e}")
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

        # Placeholder kept for backward compatibility; route uses predict_from_features
        nutrition_cols = ["calories", "carbs", "sugar", "sel", "alcool", "water"]
        for col in nutrition_cols:
            if col not in data.columns:
                data[col] = 0.0
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
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
                normalized_data[col] = (
                    normalized_data[col] - self.normalization_stats[col]["mean"]
                )
        nutrition_tensor = torch.tensor(
            normalized_data.values, dtype=torch.float32
        ).unsqueeze(0)
        with torch.no_grad():
            base_metabolisms = self.model(nutrition_tensor)
            return base_metabolisms.squeeze().tolist()

    def predict_from_features(self, features_df: pd.DataFrame):
        if self.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction model not loaded.",
            )
        # Ensure required columns and numeric types
        nutrition_cols = ["calories", "carbs", "sugar", "sel", "alcool", "water"]
        required_cols = nutrition_cols + ["sport", "pds"]
        for col in required_cols:
            if col not in features_df.columns:
                features_df[col] = 0.0
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(0)

        # Normalize using stored stats
        norm_df = features_df.copy()
        for col in nutrition_cols + ["sport"]:
            if col in self.normalization_stats:
                mean = self.normalization_stats[col]["mean"]
                std = self.normalization_stats[col]["std"] or 1.0
                norm_df[col] = (norm_df[col] - mean) / std

        weight_mean = self.normalization_stats.get("pds", {}).get("mean", 0.0)
        norm_df["pds_normalized"] = norm_df["pds"] - weight_mean

        # Tensors
        observed_weights = torch.tensor(
            norm_df["pds_normalized"].values, dtype=torch.float32
        ).unsqueeze(0)
        nutrition_tensor = torch.tensor(
            norm_df[nutrition_cols].values, dtype=torch.float32
        ).unsqueeze(0)
        sport_tensor = torch.tensor(norm_df["sport"].values, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            base_metabolisms = self.model(nutrition_tensor)
            predicted_observed_weight, w_adj_pred = reconstruct_trajectory(
                observed_weights,
                base_metabolisms,
                nutrition_tensor,
                sport_tensor,
                self.model.initial_adj_weight,
                self.normalization_stats,
            )

        # De-normalize outputs
        base_metabolisms_kcal = base_metabolisms.squeeze().cpu().numpy() * 1000.0
        w_adj_pred_actual = w_adj_pred.squeeze().cpu().numpy() + weight_mean
        actual_weight = features_df["pds"].values
        water_retention = actual_weight - w_adj_pred_actual

        return {
            "actual_weight": actual_weight.tolist(),
            "predicted_adjusted_weight": w_adj_pred_actual.tolist(),
            "water_retention": water_retention.tolist(),
            "base_metabolism_kcal": base_metabolisms_kcal.tolist(),
        }


@app.get("/api/v1/health")
def health_check():
    return {"status": "ok"}


prediction_service = PredictionService()


@app.on_event("startup")
async def startup_event():
    prediction_service.load_model()


@app.get("/api/v1/predict/latest", tags=["prediction"])
async def get_latest_prediction():
    # Use the prepared features/results file as the input source
    results_path = "data/final_results.csv"
    if not os.path.exists(results_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Results file not found. Please run train_model.py first.",
        )
    df = pd.read_csv(results_path)
    outputs = prediction_service.predict_from_features(df)
    # Return latest point plus series
    latest_idx = len(df) - 1
    return {
        "latest": {
            "actual_weight": outputs["actual_weight"][latest_idx],
            "predicted_adjusted_weight": outputs["predicted_adjusted_weight"][latest_idx],
            "water_retention": outputs["water_retention"][latest_idx],
            "base_metabolism_kcal": outputs["base_metabolism_kcal"][latest_idx],
        },
        "series": outputs,
    }


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
        with open("models/best_params.json", "r") as f:
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


@app.get("/plots", tags=["plots"])
def plots_page(request: Request):
    return templates.TemplateResponse("plots.html", {"request": request})
