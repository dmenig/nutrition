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
import pickle
from typing import Any
from fastapi import BackgroundTasks

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(auth.router, prefix="/api/v1", tags=["users"])
app.include_router(custom_foods.router, prefix="", tags=["custom-foods"])
app.include_router(sport_activities.router, prefix="", tags=["sports"])
app.include_router(food_logs.router, prefix="", tags=["food-logs"])


class PredictionService:
    model: Any = None
    model_path: str = "/app/models/model.pkl"

    def __init__(self):
        self.load_model()

    def load_model(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        print(f"Model loaded successfully from {self.model_path}")

    def predict(self, data: pd.DataFrame):
        if self.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction model not loaded.",
            )
        return self.model.predict(data).tolist()


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
