import os
import json
from typing import Tuple

import numpy as np
import pandas as pd


def _load_normalization(params_path: str) -> dict:
    with open(params_path, "r") as f:
        params = json.load(f)
    return params.get("normalization", {})


def _build_training_features(journal_csv: str, variables_csv: str) -> pd.DataFrame:
    # Use the same path the trainer uses to avoid DB dependencies
    from build_features import main as build_features_main

    features_df = build_features_main(journal_path=journal_csv, variables_path=variables_csv)
    # Ensure required columns exist and numeric
    nutrition_cols = ["calories", "carbs", "sugar", "sel", "alcool", "water"]
    for col in nutrition_cols + ["pds", "sport"]:
        if col not in features_df.columns:
            features_df[col] = 0.0
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(0)
    return features_df


def _normalize_features(df: pd.DataFrame, normalization: dict) -> Tuple[pd.DataFrame, float]:
    nutrition_cols = ["calories", "carbs", "sugar", "sel", "alcool", "water"]
    norm_df = df.copy()
    for col in nutrition_cols + ["sport"]:
        mean = float(normalization.get(col, {}).get("mean", 0.0))
        std = float(normalization.get(col, {}).get("std", 1.0) or 1.0)
        norm_df[col] = (norm_df[col] - mean) / std
    weight_mean = float(normalization.get("pds", {}).get("mean", 0.0))
    norm_df["pds_normalized"] = norm_df["pds"] - weight_mean
    return norm_df, weight_mean


def _torch_forward(norm_df: pd.DataFrame, normalization: dict) -> Tuple[np.ndarray, np.ndarray]:
    import torch
    from train_model import FinalModel, reconstruct_trajectory

    nutrition_cols = ["calories", "carbs", "sugar", "sel", "alcool", "water"]

    observed_weights = torch.tensor(norm_df["pds_normalized"].values, dtype=torch.float32).unsqueeze(0)
    nutrition_data = torch.tensor(norm_df[nutrition_cols].values, dtype=torch.float32).unsqueeze(0)
    sport_data = torch.tensor(norm_df["sport"].values, dtype=torch.float32).unsqueeze(0)

    # Load Torch model
    models_dir = os.path.join(os.getcwd(), "models")
    pth_path = os.path.join(models_dir, "recurrent_model.pth")
    sd = torch.load(pth_path, map_location="cpu")
    model = FinalModel(nutrition_data.shape[2], initial_weight_guess=float(norm_df["pds_normalized"].iloc[:5].mean()))
    model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        base_metabolisms = model(nutrition_data)
        w_pred, w_adj_pred = reconstruct_trajectory(
            observed_weights,
            base_metabolisms,
            nutrition_data,
            sport_data,
            model.initial_adj_weight,
            normalization,
        )
    return base_metabolisms.squeeze().cpu().numpy(), w_adj_pred.squeeze().cpu().numpy()


def _numpy_forward(norm_df: pd.DataFrame, normalization: dict) -> Tuple[np.ndarray, np.ndarray]:
    from app.np_infer import load_numpy_weights, NumpyFinalModel, reconstruct_trajectory_numpy

    nutrition_cols = ["calories", "carbs", "sugar", "sel", "alcool", "water"]

    # Inputs
    obs_np = norm_df["pds_normalized"].values.astype("float32")[None, :]
    nut_np = norm_df[nutrition_cols].values.astype("float32")[None, :, :]
    sport_np = norm_df["sport"].values.astype("float32")[None, :]

    models_dir = os.path.join(os.getcwd(), "models")
    npz_path = os.path.join(models_dir, "recurrent_model_np.npz")
    weights = load_numpy_weights(npz_path)

    gru_w = weights.get("gru.weight_hh_l0")
    head_w = weights.get("head.0.weight")
    hidden_size = gru_w.shape[1]
    input_size = head_w.shape[1] - hidden_size
    num_layers = sum(1 for k in weights.keys() if k.startswith("gru.weight_ih_l"))

    np_model = NumpyFinalModel(weights, input_size, hidden_size, num_layers)
    base_metabolisms = np_model.forward(nut_np)
    _, w_adj_pred = reconstruct_trajectory_numpy(
        obs_np, base_metabolisms, nut_np, sport_np, float(np_model.initial_adj_weight), normalization
    )
    return base_metabolisms.squeeze(), w_adj_pred.squeeze()


def main() -> None:
    cwd = os.getcwd()
    models_dir = os.path.join(cwd, "models")
    params_path = os.path.join(models_dir, "best_params.json")
    journal_csv = os.path.join(cwd, "data", "processed_journal.csv")
    variables_csv = os.path.join(cwd, "data", "variables.csv")

    assert os.path.exists(params_path), f"Missing params: {params_path}"
    assert os.path.exists(os.path.join(models_dir, "recurrent_model.pth")), "Missing Torch model .pth"
    assert os.path.exists(os.path.join(models_dir, "recurrent_model_np.npz")), "Missing NumPy weights .npz"
    assert os.path.exists(journal_csv) and os.path.exists(variables_csv), "Missing training CSVs in data/"

    normalization = _load_normalization(params_path)
    feats = _build_training_features(journal_csv, variables_csv)
    norm_df, weight_mean = _normalize_features(feats, normalization)

    base_t, w_adj_t = _torch_forward(norm_df, normalization)
    base_n, w_adj_n = _numpy_forward(norm_df, normalization)

    # Convert base metabolisms to kcal/day for human-friendly diffs
    base_t_kcal = base_t * 1000.0
    base_n_kcal = base_n * 1000.0
    w_adj_t_abs = w_adj_t + weight_mean
    w_adj_n_abs = w_adj_n + weight_mean

    def _summ(name: str, a: np.ndarray, b: np.ndarray) -> None:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        l2 = float(np.linalg.norm(a - b))
        max_abs = float(np.max(np.abs(a - b)))
        print(f"{name}: l2={l2:.6f} max_abs={max_abs:.6f}")

    print("=== Backend Parity Check (Torch vs NumPy) ===")
    print(f"Sequence length: {len(base_t)}")
    _summ("base_metabolism_kcal", base_t_kcal, base_n_kcal)
    _summ("W_adj_pred", w_adj_t_abs, w_adj_n_abs)

    # Print a couple of sample rows to eyeball
    for idx in [0, len(base_t) // 2, len(base_t) - 1]:
        print(
            f"t={idx:4d} | M_base: torch={base_t_kcal[idx]:.3f} vs numpy={base_n_kcal[idx]:.3f} | "
            f"W_adj: torch={w_adj_t_abs[idx]:.3f} vs numpy={w_adj_n_abs[idx]:.3f}"
        )


if __name__ == "__main__":
    main()


