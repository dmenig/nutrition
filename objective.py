import pandas as pd
import optuna

from weight_model import WeightModel


def run_model_simulation(
    params: dict, features_df: pd.DataFrame, f_water_model_params: dict
) -> pd.DataFrame:
    """
    Runs the full simulation using the WeightModel.

    Args:
        params (dict): Model parameters.
        features_df (pd.DataFrame): DataFrame with features.
        f_water_model_params (dict): Parameters for the water retention model.

    Returns:
        pd.DataFrame: DataFrame with model outputs.
    """
    model = WeightModel(
        initial_M_base=params["initial_M_base"],
        alpha=params["alpha"],
        K_cal_kg=params["K_cal_kg"],
        look_back_window=params["look_back_window"],
        f_water_model_params=f_water_model_params,
    )

    # First pass to train the metabolism model
    results_df = model.run(features_df.copy())

    # Second pass to train the water retention model
    model.fit_water_model(features_df.copy(), results_df)
    results_df = model.run(features_df.copy())

    if "pds" in features_df.columns:
        results_df = pd.merge(
            results_df, features_df[["pds"]], left_index=True, right_index=True
        )

    return results_df


def calculate_loss(
    results_df: pd.DataFrame, w_meta: float, w_water: float, w_pred: float
) -> float:
    """
    Calculates the final objective value.
    Args:
        results_df (pd.DataFrame): DataFrame from run_model_simulation.
        w_meta (float): Weight for the metabolism change penalty.
        w_water (float): Weight for the water retention regularization.
        w_pred (float): Weight for the prediction error.
    Returns:
        float: The total calculated loss.
    """
    l_meta = results_df["M_base"].diff().pow(2).mean()
    l_water = results_df["WR_t"].pow(2).mean()
    l_pred = (results_df["W_pred_t"] - results_df["W_obs_t"]).pow(2).mean()
    total_loss = (w_meta * l_meta) + (w_water * l_water) + (w_pred * l_pred)
    return total_loss


def objective(
    trial: optuna.Trial,
    features_df: pd.DataFrame,
    w_meta: float,
    w_water: float,
    w_pred: float,
    f_water_model_params: dict,
) -> float:
    """
    Optuna objective function.
    Args:
        trial (optuna.Trial): Optuna trial object.
        features_df (pd.DataFrame): DataFrame with features.
        w_meta (float): Weight for the metabolism change penalty.
        w_water (float): Weight for the water retention regularization.
        w_pred (float): Weight for the prediction error.
        f_water_model_params (dict): Parameters for the water retention model.
    Returns:
        float: The total calculated loss.
    """
    params = {
        "initial_M_base": trial.suggest_float("initial_M_base", 1800, 2200),
        "alpha": trial.suggest_float("alpha", 0.01, 0.2),
        "look_back_window": trial.suggest_int("look_back_window", 3, 14),
        "K_cal_kg": 7700,  # Fixed value
    }

    results_df = run_model_simulation(params, features_df, f_water_model_params)
    total_loss = calculate_loss(results_df, w_meta, w_water, w_pred)

    return total_loss
