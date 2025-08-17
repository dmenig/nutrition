# Set low-memory/thread env before importing libraries that may initialize BLAS backends
import os as _early_os
_early_os.environ.setdefault("OMP_NUM_THREADS", "1")
_early_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_early_os.environ.setdefault("MKL_NUM_THREADS", "1")
_early_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
_early_os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
_early_os.environ.setdefault("MALLOC_ARENA_MAX", "2")

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from app.db.database import engine, Base, get_db, SessionLocal
from app.db.models import DailySummary, FoodLog, SportActivity, User, WeightLog
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
from threading import Lock, Thread
from fastapi import BackgroundTasks
import gc
# Defer heavy imports from `train_model` until needed to keep memory/CPU footprint low
import json  # Added import for json
import pathlib
from sqlalchemy import func
from datetime import datetime, timezone, timedelta
from app.services.summary import upsert_daily_summary

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
    np_model: Any = None
    model_path: str = "models/recurrent_model.pth"
    params_path: str = "models/best_params.json"
    npz_path: str = "models/recurrent_model_np.npz"
    normalization_stats: dict = {}  # Initialize normalization_stats

    def __init__(self):
        # Lazy-load the model to avoid importing torch until needed
        pass

    def _resolve_first_existing(self, candidates):
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def load_model(self):
        # NumPy-only deployment: load normalization stats if present; skip Torch entirely
        try:
            params_path = self._resolve_first_existing(
                [self.params_path, "/app/best_params.json", "/app/models/best_params.json"]
            )
            with open(params_path, "r") as f:
                params = json.load(f)
                self.normalization_stats = params.get("normalization", {})
        except Exception:
            # Keep defaults; identity normalization will be used
            self.normalization_stats = {}
        print("NumPy-only deployment: load_model skipped Torch load.")

    def _export_numpy_weights(self, npz_out_path: str) -> None:
        # NumPy-only deployment: exporting from Torch is disabled
        return

    def _should_use_numpy(self) -> bool:
        """Return True if NumPy backend should be used based on env and file mtimes.

        - PREDICT_BACKEND=torch forces Torch
        - PREDICT_BACKEND=numpy forces NumPy (if file exists)
        - Otherwise, prefer NumPy only if the .npz exists and is at least as new as the .pth
        """
        backend = os.environ.get("PREDICT_BACKEND", "auto").lower()
        if backend == "torch":
            return False
        if not os.path.exists(self.npz_path):
            return False
        if backend == "numpy":
            return True
        # Compare mtimes: use numpy only if npz is >= pth
        pth_candidates = [
            self.model_path,
            "/app/recurrent_model.pth",
            "/app/models/recurrent_model.pth",
        ]
        pth_path = next((p for p in pth_candidates if os.path.exists(p)), None)
        try:
            if pth_path is not None:
                return os.path.getmtime(self.npz_path) >= os.path.getmtime(pth_path)
        except Exception:
            pass
        # If we cannot compare, default to NumPy since file exists
        return True

    def _ensure_normalization_stats_loaded(self) -> None:
        """Load normalization stats from best_params.json if not already loaded.

        This is required for both torch and numpy inference paths. On Render, we
        often prefer the numpy path to avoid torch imports; in that case this
        helper ensures self.normalization_stats is populated.
        """
        # If already loaded, nothing to do
        if isinstance(self.normalization_stats, dict) and self.normalization_stats:
            return
        params_candidates = [
            self.params_path,
            "/app/best_params.json",
            "/app/models/best_params.json",
        ]
        params_path = next((p for p in params_candidates if os.path.exists(p)), None)
        if params_path is not None:
            try:
                with open(params_path, "r") as f:
                    params = json.load(f)
                    self.normalization_stats = params.get("normalization", {})
            except Exception:
                # Fall back to identity stats below
                self.normalization_stats = {}
        # Ensure required keys exist with identity transforms
        for key in ["calories", "carbs", "sugar", "sel", "alcool", "water", "sport"]:
            if key not in self.normalization_stats:
                self.normalization_stats[key] = {"mean": 0.0, "std": 1.0}
        if "pds" not in self.normalization_stats:
            # Use a realistic human weight mean to produce absolute weights when params are missing
            self.normalization_stats["pds"] = {"mean": 70.0}

    def predict(self, data: pd.DataFrame):
        # NumPy-only inference
        if self.np_model is None:
            from app.np_infer import load_numpy_weights, NumpyFinalModel  # noqa: WPS433
            weights = load_numpy_weights(self.npz_path)
            # Infer architecture sizes from saved shapes
            head_w = weights.get("head.0.weight")
            gru_w = weights.get("gru.weight_hh_l0")
            if head_w is None or gru_w is None:
                raise RuntimeError("NP weights missing required tensors")
            hidden_size = gru_w.shape[1]
            input_size = head_w.shape[1] - hidden_size
            num_layers = sum(1 for k in weights.keys() if k.startswith("gru.weight_ih_l"))
            self.np_model = NumpyFinalModel(weights, input_size, hidden_size, num_layers)
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
        features = normalized_data.values.astype("float32")[None, :, :]
        base_metabolisms = self.np_model.forward(features)
        return base_metabolisms.squeeze().tolist()

    def predict_from_features(self, features_df: pd.DataFrame):
        # NumPy-only inference
        if self.np_model is None:
            from app.np_infer import load_numpy_weights, NumpyFinalModel  # noqa: WPS433
            weights = load_numpy_weights(self.npz_path)
            gru_w = weights.get("gru.weight_hh_l0")
            head_w = weights.get("head.0.weight")
            hidden_size = gru_w.shape[1]
            input_size = head_w.shape[1] - hidden_size
            num_layers = sum(1 for k in weights.keys() if k.startswith("gru.weight_ih_l"))
            self.np_model = NumpyFinalModel(weights, input_size, hidden_size, num_layers)
        # Ensure normalization params are loaded
        self._ensure_normalization_stats_loaded()
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

        weight_mean = float(self.normalization_stats.get("pds", {}).get("mean", 0.0) or 0.0)
        norm_df["pds_normalized"] = norm_df["pds"] - weight_mean

        # NumPy path
        if self.np_model is not None:
            obs_np = norm_df["pds_normalized"].values.astype("float32")[None, :]
            nut_np = norm_df[nutrition_cols].values.astype("float32")[None, :, :]
            sport_np = norm_df["sport"].values.astype("float32")[None, :]
            base_metabolisms = self.np_model.forward(nut_np)
            from app.np_infer import reconstruct_trajectory_numpy  # noqa: WPS433
            predicted_observed_weight, w_adj_pred = reconstruct_trajectory_numpy(
                obs_np,
                base_metabolisms,
                nut_np,
                sport_np,
                float(self.np_model.initial_adj_weight),
                self.normalization_stats,
            )
        else:
            raise HTTPException(status_code=503, detail="NumPy weights not loaded")

        # De-normalize outputs
        # Convert results to numpy uniformly
        import numpy as _np
        base_metabolisms_np = base_metabolisms.squeeze()
        w_adj_np = w_adj_pred.squeeze()
        base_metabolisms_kcal = base_metabolisms_np * 1000.0
        w_adj_pred_actual = w_adj_np + weight_mean
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


# In-memory plot cache (full DataFrame, sliced per request)
_PLOT_CACHE_LOCK: Lock = Lock()
_PLOT_CACHE_DF: pd.DataFrame | None = None


@app.on_event("startup")
async def startup_event():
    # Warm the plot cache in the background so first user hits are fast
    def _warm_cache() -> None:
        try:
            # Compute once; get_plot_data will populate the cache
            _ = get_plot_data(last_n=None)
        except Exception:
            # Best effort; failures are non-fatal
            pass

    try:
        t = Thread(target=_warm_cache, daemon=True)
        t.start()
    except Exception:
        pass
    return None


@app.get("/api/v1/predict/latest", tags=["prediction"])
async def get_latest_prediction():
    """Return latest model outputs using DB-derived features only (no fallbacks)."""
    db = SessionLocal()
    try:
        # Aggregate minimal features per day from DB
        date_expr_fl = func.coalesce(FoodLog.logged_date, func.date(FoodLog.logged_at))
        food_rows = (
            db.query(
                date_expr_fl.label("date"),
                func.coalesce(func.sum(FoodLog.calories), 0.0).label("calories"),
                func.coalesce(func.sum(FoodLog.carbs), 0.0).label("carbs"),
            )
            .group_by(date_expr_fl)
            .all()
        )
        date_expr_sa = func.coalesce(SportActivity.logged_date, func.date(SportActivity.logged_at))
        sport_rows = (
            db.query(
                date_expr_sa.label("date"),
                func.coalesce(func.sum(SportActivity.calories_expended), 0.0).label("sport"),
            )
            .group_by(date_expr_sa)
            .all()
        )
    finally:
        db.close()

    dates = sorted({r.date for r in food_rows} | {r.date for r in sport_rows})
    if not dates:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No DB data available for prediction.",
        )
    food_by_date = {r.date: r for r in food_rows}
    sport_by_date = {r.date: r for r in sport_rows}
    records: list[dict] = []
    for d in dates:
        fr = food_by_date.get(d)
        sr = sport_by_date.get(d)
        records.append(
            {
                "date": d,
                "calories": float(getattr(fr, "calories", 0.0) or 0.0),
                "carbs": float(getattr(fr, "carbs", 0.0) or 0.0),
                "sugar": 0.0,
                "sel": 0.0,
                "alcool": 0.0,
                "water": 0.0,
                "sport": float(getattr(sr, "sport", 0.0) or 0.0),
                # Observed weights unknown in DB; set to 0 for model reconstruction
                "pds": 0.0,
            }
        )
    features_df = pd.DataFrame(records)
    outputs = prediction_service.predict_from_features(features_df)
    latest_idx = len(features_df) - 1
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


def get_plot_data(last_n: int | None = None):
    # Construct plot data strictly from the database (no CSVs in production)
    expected_cols = [
        "W_obs",
        "W_adj_pred",
        "M_base",
        "calories",
        "sport",
    ]

    # Serve from cache if available
    global _PLOT_CACHE_DF
    try:
        with _PLOT_CACHE_LOCK:
            cached_df = _PLOT_CACHE_DF
        if cached_df is not None and not cached_df.empty:
            df = cached_df
            if isinstance(last_n, int) and last_n > 0 and len(df) > last_n:
                return df.tail(last_n).reset_index(drop=True)
            return df
    except Exception:
        # On any cache access issue, fall back to recompute
        pass

    def _build_from_daily_summaries() -> pd.DataFrame:
        db = SessionLocal()
        try:
            # Unscoped: aggregate across all users by date for public plots
            rows = db.query(DailySummary).order_by(DailySummary.date.asc()).all()
            if not rows:
                return pd.DataFrame(columns=expected_cols)
            df = pd.DataFrame([
                {
                    "date": r.date,
                    "calories": r.calories_total or 0.0,
                    "sport": r.sport_calories_total or 0.0,
                }
                for r in rows
            ])
            # Sum across users per date
            df = df.groupby("date", as_index=False).sum(numeric_only=True)
            # Join observed weights if any
            date_expr_w = func.coalesce(WeightLog.logged_date, func.date(WeightLog.logged_at))
            w_rows = (
                db.query(date_expr_w.label("date"), func.avg(WeightLog.weight_kg).label("W_obs"))
                .group_by(date_expr_w)
                .all()
            )
            if w_rows:
                w_df = pd.DataFrame([{"date": r.date, "W_obs": float(r.W_obs or 0)} for r in w_rows])
                df = df.merge(w_df, on="date", how="left")
            else:
                df["W_obs"] = pd.Series(dtype=float)
            df["W_adj_pred"] = pd.Series(dtype=float)
            # Keep date for downstream time_index construction; do not slice yet
            return df
        finally:
            db.close()

    def _build_from_db_on_the_fly() -> pd.DataFrame:
        """Aggregate directly from FoodLog and SportActivity per day when no summaries/CSV exist."""
        db = SessionLocal()
        try:
            # collect all distinct dates from both tables
            food_dates = [d[0] for d in db.query(FoodLog.logged_date).distinct().all()]
            sport_dates = [d[0] for d in db.query(SportActivity.logged_date).distinct().all()]
            date_expr_w = func.coalesce(WeightLog.logged_date, func.date(WeightLog.logged_at))
            weight_dates = [d[0] for d in db.query(date_expr_w).distinct().all()]
            all_dates = sorted({*food_dates, *sport_dates, *weight_dates})
            if not all_dates:
                return pd.DataFrame(columns=expected_cols)
            records = []
            for d in all_dates:
                day_start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
                day_end = day_start + timedelta(days=1)
                cal, prot, carb, fat = (
                    db.query(
                        func.coalesce(func.sum(FoodLog.calories), 0.0),
                        func.coalesce(func.sum(FoodLog.protein), 0.0),
                        func.coalesce(func.sum(FoodLog.carbs), 0.0),
                        func.coalesce(func.sum(FoodLog.fat), 0.0),
                    )
                    .filter(FoodLog.logged_date == d)
                    .first()
                )
                sport_total = (
                    db.query(func.coalesce(func.sum(SportActivity.calories_expended), 0.0))
                    .filter(SportActivity.logged_date == d)
                    .scalar()
                    or 0.0
                )
                # Observed weight: average per day across all users for public plots
                date_expr_w = func.coalesce(WeightLog.logged_date, func.date(WeightLog.logged_at))
                w_obs = (
                    db.query(func.avg(WeightLog.weight_kg))
                    .filter(date_expr_w == d)
                    .scalar()
                )
                records.append(
                    {
                        "date": d,
                        "calories": float(cal or 0.0),
                        "sport": float(sport_total or 0.0),
                        "M_base": 2500.0,
                        "W_obs": float(w_obs) if w_obs is not None else None,
                    }
                )
            df = pd.DataFrame(records)
            df["W_adj_pred"] = pd.Series(dtype=float)
            return df
        finally:
            db.close()

    def _build_features_from_db() -> pd.DataFrame:
        """Build minimal features directly from DB for model usage (no CSV)."""
        db = SessionLocal()
        try:
            # Aggregate per day calories and carbs; set other nutrition features to 0
            date_expr_fl = func.coalesce(FoodLog.logged_date, func.date(FoodLog.logged_at))
            food_rows = (
                db.query(
                    date_expr_fl.label("date"),
                    func.coalesce(func.sum(FoodLog.calories), 0.0).label("calories"),
                    func.coalesce(func.sum(FoodLog.carbs), 0.0).label("carbs"),
                )
                .group_by(date_expr_fl)
                .all()
            )
            date_expr_sa = func.coalesce(SportActivity.logged_date, func.date(SportActivity.logged_at))
            sport_rows = (
                db.query(
                    date_expr_sa.label("date"),
                    func.coalesce(func.sum(SportActivity.calories_expended), 0.0).label("sport"),
                )
                .group_by(date_expr_sa)
                .all()
            )
            dates = sorted({r.date for r in food_rows} | {r.date for r in sport_rows})
            if not dates:
                return pd.DataFrame(columns=["date", "calories", "carbs", "sugar", "sel", "alcool", "water", "sport", "pds"])  # empty
            food_by_date = {r.date: r for r in food_rows}
            sport_by_date = {r.date: r for r in sport_rows}
            records: list[dict] = []
            for d in dates:
                fr = food_by_date.get(d)
                sr = sport_by_date.get(d)
                records.append(
                    {
                        "date": d,
                        "calories": float(getattr(fr, "calories", 0.0) or 0.0),
                        "carbs": float(getattr(fr, "carbs", 0.0) or 0.0),
                        "sugar": 0.0,
                        "sel": 0.0,
                        "alcool": 0.0,
                        "water": 0.0,
                        "sport": float(getattr(sr, "sport", 0.0) or 0.0),
                        # Observed weight unknown in DB; use 0 to allow model to run
                        "pds": 0.0,
                    }
                )
            return pd.DataFrame(records)
        finally:
            db.close()

    # Priority (DB-only):
    # - Normal: features from DB + model -> daily summaries -> direct DB agg
    # - Lightweight: daily summaries -> direct DB agg
    # Always try model using DB-derived features first (DL model path)
    features_df = _build_features_from_db()
    df = pd.DataFrame()
    if features_df is not None and not features_df.empty:
        try:
            outputs = prediction_service.predict_from_features(features_df)
            df = pd.DataFrame(
                {
                    "date": features_df["date"],
                    # Use observed weights from DB if available; fall back to model input only if non-zero
                    "W_obs": outputs.get("actual_weight", []),
                    "W_adj_pred": outputs.get("predicted_adjusted_weight", []),
                    "M_base": outputs.get("base_metabolism_kcal", []),
                    "calories": features_df.get("calories", 0),
                    "sport": features_df.get("sport", 0),
                }
            )
            # Replace W_obs with true observed weights from DB when present (by date)
            try:
                db_obs = SessionLocal()
                try:
                    date_expr_w = func.coalesce(WeightLog.logged_date, func.date(WeightLog.logged_at))
                    w_rows = (
                        db_obs.query(date_expr_w.label("date"), func.avg(WeightLog.weight_kg).label("W_obs"))
                        .group_by(date_expr_w)
                        .all()
                    )
                    if w_rows:
                        w_df = pd.DataFrame([{"date": r.date, "W_obs": float(r.W_obs or 0)} for r in w_rows])
                        df = df.merge(w_df, on="date", how="left", suffixes=("", "_db"))
                        # Prefer DB observed weights when available; otherwise keep model input values
                        if "W_obs_db" in df.columns:
                            df["W_obs"] = df["W_obs_db"].where(pd.notnull(df["W_obs_db"]), df["W_obs"]).astype(float)
                            df.drop(columns=["W_obs_db"], inplace=True)
                finally:
                    db_obs.close()
            except Exception:
                pass
        except Exception:
            df = pd.DataFrame()
    # No fallbacks when enforcing DL-only path
    # Safety nets only if DL path failed or produced empty
    if df.empty:
        df = _build_from_daily_summaries()
    if df.empty:
        df = _build_from_db_on_the_fly()

    # Ensure expected columns exist
    for col_name in expected_cols:
        if col_name not in df.columns:
            df[col_name] = pd.Series(dtype=float)
    # Keep only relevant columns plus date if present
    keep_cols = [c for c in (["date"] + expected_cols) if c in df.columns]
    df = df[keep_cols]
    # Coerce numeric types. Do NOT fill missing weights with zeros; keep NaN to signal "unknown".
    for col_name in ["calories", "sport", "M_base"]:
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0)
    for col_name in ["W_obs", "W_adj_pred"]:
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

    # Build time_index from date if available; otherwise fall back to sequential index
    if "date" in df.columns:
        try:
            dt = pd.to_datetime(df["date"], errors="coerce")
            # Drop rows with invalid dates to avoid epoch 1970 artifacts
            valid_mask = dt.notna()
            df = df.loc[valid_mask].copy()
            dt = dt.loc[valid_mask]
            # Convert to epoch milliseconds
            # Using view("int64") is compatible with modern pandas
            df["time_index"] = (dt.view("int64") // 1_000_000).astype("int64")
        except Exception:
            # As a conservative fallback, keep a monotonically increasing daily index anchored to today
            start_ms = int(pd.Timestamp.utcnow().normalize().value // 1_000_000)
            one_day_ms = 24 * 60 * 60 * 1000
            df["time_index"] = [start_ms + i * one_day_ms for i in range(len(df))]
    else:
        # If we truly have no dates, synthesize a daily timeline anchored to today
        start_ms = int(pd.Timestamp.utcnow().normalize().value // 1_000_000)
        one_day_ms = 24 * 60 * 60 * 1000
        df["time_index"] = [start_ms + i * one_day_ms for i in range(len(df))]

    # Try to load normalization parameters; if missing, use identity transform
    calories_mean = 0.0
    calories_std = 1.0
    sport_mean = 0.0
    sport_std = 1.0
    params_candidates = [
        "models/best_params.json",
        "/app/models/best_params.json",
        "/app/best_params.json",
    ]
    params_path = next((p for p in params_candidates if os.path.exists(p)), None)
    if params_path is not None:
        try:
            with open(params_path, "r") as f:
                params = json.load(f)
                normalization_stats = params.get("normalization", {})
                calories_mean = normalization_stats.get("calories", {}).get("mean", 0.0)
                calories_std = normalization_stats.get("calories", {}).get("std", 1.0) or 1.0
                sport_mean = normalization_stats.get("sport", {}).get("mean", 0.0)
                sport_std = normalization_stats.get("sport", {}).get("std", 1.0) or 1.0
        except Exception:
            # If params are unreadable, keep identity transforms
            pass

    # IMPORTANT: values in df["calories"] and df["sport"] are already absolute (kcal)
    # because they come directly from the DB aggregations. Do not apply de-normalization
    # intended for z-scored training tensors. Use pass-through here to avoid inflated values
    # that would not match the PNG exported from training results.
    df["calories_unnorm"] = pd.to_numeric(df["calories"], errors="coerce").fillna(0)
    df["sport_unnorm"] = pd.to_numeric(df["sport"], errors="coerce").fillna(0)
    df["C_exp_t"] = df["M_base"].fillna(0) + df["sport_unnorm"]

    # Ensure sorted by time for slicing
    if "time_index" in df.columns:
        df = df.sort_values(by=["time_index"]).reset_index(drop=True)
    # Populate cache with the full DataFrame before any slicing
    try:
        with _PLOT_CACHE_LOCK:
            _PLOT_CACHE_DF = df.copy()
    except Exception:
        pass

    # Optional slicing to last N days/points for this response
    if isinstance(last_n, int) and last_n > 0 and len(df) > last_n:
        df = df.tail(last_n).reset_index(drop=True)
    return df


## moved to app.services.summary


@app.get("/api/v1/plots/debug", tags=["plots"])
def plots_debug():
    from urllib.parse import urlparse
    db_url = os.environ.get("DATABASE_URL", "")
    parsed = urlparse(db_url) if db_url else None
    host = parsed.hostname if parsed else None
    debug = {
        "cwd": str(pathlib.Path.cwd()),
        "db_host": host,
        "db_daily_summaries": 0,
        "db_food_days": 0,
        "db_sport_days": 0,
        "final_rows": 0,
        "final_cols": [],
    }
    try:
        db = SessionLocal()
        try:
            debug["db_daily_summaries"] = int(db.query(func.count(DailySummary.id)).scalar() or 0)
            dfl = func.coalesce(FoodLog.logged_date, func.date(FoodLog.logged_at))
            dsa = func.coalesce(SportActivity.logged_date, func.date(SportActivity.logged_at))
            debug["db_food_days"] = int(db.query(func.count(func.distinct(dfl))).scalar() or 0)
            debug["db_sport_days"] = int(db.query(func.count(func.distinct(dsa))).scalar() or 0)
        finally:
            db.close()
    except Exception as e:
        debug["db_error"] = str(e)
    try:
        df_final = get_plot_data()
        debug["final_rows"] = int(len(df_final))
        debug["final_cols"] = list(df_final.columns)
    except Exception as e:
        debug["final_error"] = str(e)
    # Extra diagnostics for model/params presence and DL path
    try:
        debug["has_npz"] = os.path.exists(prediction_service.npz_path)
        debug["has_params"] = os.path.exists(prediction_service.params_path) or os.path.exists("/app/models/best_params.json")
        # Build a tiny features DF like /predict/latest uses
        db = SessionLocal()
        try:
            date_expr_fl = func.coalesce(FoodLog.logged_date, func.date(FoodLog.logged_at))
            food_rows = (
                db.query(
                    date_expr_fl.label("date"),
                    func.coalesce(func.sum(FoodLog.calories), 0.0).label("calories"),
                    func.coalesce(func.sum(FoodLog.carbs), 0.0).label("carbs"),
                )
                .group_by(date_expr_fl)
                .order_by(date_expr_fl)
                .all()
            )
            date_expr_sa = func.coalesce(SportActivity.logged_date, func.date(SportActivity.logged_at))
            sport_rows = (
                db.query(
                    date_expr_sa.label("date"),
                    func.coalesce(func.sum(SportActivity.calories_expended), 0.0).label("sport"),
                )
                .group_by(date_expr_sa)
                .order_by(date_expr_sa)
                .all()
            )
        finally:
            db.close()
        dates = sorted({r.date for r in food_rows} | {r.date for r in sport_rows})
        if dates:
            food_by_date = {r.date: r for r in food_rows}
            sport_by_date = {r.date: r for r in sport_rows}
            # Use last 14 days for a quick probe
            tail_dates = dates[-14:]
            records: list[dict] = []
            for d in tail_dates:
                fr = food_by_date.get(d)
                sr = sport_by_date.get(d)
                records.append(
                    {
                        "date": d,
                        "calories": float(getattr(fr, "calories", 0.0) or 0.0),
                        "carbs": float(getattr(fr, "carbs", 0.0) or 0.0),
                        "sugar": 0.0,
                        "sel": 0.0,
                        "alcool": 0.0,
                        "water": 0.0,
                        "sport": float(getattr(sr, "sport", 0.0) or 0.0),
                        "pds": 0.0,
                    }
                )
            feat = pd.DataFrame(records)
            try:
                outs = prediction_service.predict_from_features(feat)
                debug["dl_ok"] = True
                debug["dl_series_keys"] = list(outs.keys())
                debug["dl_w_adj_len"] = int(len(outs.get("predicted_adjusted_weight", [])))
                debug["dl_m_base_len"] = int(len(outs.get("base_metabolism_kcal", [])))
                # Show min/max to catch all-zero
                import numpy as _np
                w_adj = _np.asarray(outs.get("predicted_adjusted_weight", []), dtype=float)
                m_base = _np.asarray(outs.get("base_metabolism_kcal", []), dtype=float)
                if w_adj.size:
                    debug["dl_w_adj_minmax"] = [float(_np.nanmin(w_adj)), float(_np.nanmax(w_adj))]
                if m_base.size:
                    debug["dl_m_base_minmax"] = [float(_np.nanmin(m_base)), float(_np.nanmax(m_base))]
            except Exception as ee:
                debug["dl_ok"] = False
                debug["dl_error"] = str(ee)
        else:
            debug["dl_ok"] = False
            debug["dl_error"] = "no dates in DB"
    except Exception as e2:
        debug["dl_diag_error"] = str(e2)
    return debug


@app.get("/api/v1/plots/weight", response_model=WeightPlotResponse, tags=["plots"])
def get_weight_plot_data(days: int | None = None):
    df = get_plot_data(last_n=days)
    # Enforce presence of observed weights; no fallback
    # Only include observed weights that are non-null and non-zero
    w_obs = [
        {"time_index": row["time_index"], "value": float(row["W_obs"])}
        for _, row in df.iterrows()
        if pd.notnull(row.get("W_obs")) and float(row.get("W_obs", 0) or 0) != 0.0
    ]
    w_adj = [
        {"time_index": row["time_index"], "value": float(row["W_adj_pred"])}
        for _, row in df.iterrows()
        if pd.notnull(row.get("W_adj_pred"))
    ]
    if not w_obs:
        raise HTTPException(status_code=404, detail="Observed weights unavailable in DB")
    return WeightPlotResponse(W_obs=w_obs, W_adj_pred=w_adj)


@app.get(
    "/api/v1/plots/metabolism", response_model=MetabolismPlotResponse, tags=["plots"]
)
def get_metabolism_plot_data(days: int | None = None):
    df = get_plot_data(last_n=days)
    m_values = [float(v) for v in df["M_base"].tolist() if pd.notnull(v)]
    # If metabolism is a flat placeholder at 2500 across all points, return empty to allow client fallback
    m_series: list[dict] = []
    if not (len(m_values) > 0 and all(abs(v - 2500.0) < 1e-6 for v in m_values)):
        m_series = [
            {"time_index": row["time_index"], "value": float(row["M_base"])}
            for _, row in df.iterrows()
            if pd.notnull(row.get("M_base"))
        ]
    return MetabolismPlotResponse(M_base=m_series)


@app.get(
    "/api/v1/plots/energy-balance",
    response_model=EnergyBalancePlotResponse,
    tags=["plots"],
)
def get_energy_balance_plot_data(days: int | None = None):
    df = get_plot_data(last_n=days)
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


# Alias without "/plots" to avoid 404s from clients using the documented slug only
@app.get(
    "/api/v1/energy-balance",
    response_model=EnergyBalancePlotResponse,
    tags=["plots"],
)
def get_energy_balance_plot_data_alias(days: int | None = None):
    return get_energy_balance_plot_data(days=days)


@app.get("/plots", tags=["plots"])
def plots_page(request: Request):
    return templates.TemplateResponse("plots.html", {"request": request})


# Optional admin utility to (re)build plot data on-demand in deployed envs
@app.post("/api/v1/plots/rebuild", tags=["plots"])
def rebuild_plots_data():
    """Recompute plot data once and refresh the in-memory cache."""
    global _PLOT_CACHE_DF
    try:
        # Clear and rebuild cache
        try:
            with _PLOT_CACHE_LOCK:
                _PLOT_CACHE_DF = None
        except Exception:
            pass
        df = get_plot_data(last_n=None)
        rows = int(len(df))
        return {"status": "ok", "rows": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild plots data: {e}")
