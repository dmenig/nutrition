import optuna
import pandas as pd
import numpy as np
from weight_model import WeightModel
from data_processor import load_and_process_data
from sklearn.metrics import mean_squared_error


def objective(trial: optuna.Trial, data_df: pd.DataFrame):
    """
    Optuna objective function to tune the WeightModel.

    Args:
        trial (optuna.Trial): A trial object from Optuna.
        data_df (pd.DataFrame): The preprocessed DataFrame containing all features.

    Returns:
        float: The combined loss to be minimized.
    """
    # 1. Suggest Hyperparameters for WeightModel
    K_cal_kg = trial.suggest_float("K_cal_kg", 7000, 8500)
    initial_M_base = trial.suggest_float("initial_M_base", 1500, 3000)
    alpha = trial.suggest_float(
        "alpha", 0.01, 0.5, log=True
    )  # New: alpha for EMA smoothing
    look_back_window = trial.suggest_int("look_back_window", 3, 14)

    # 2. Suggest Hyperparameters for WaterRetentionModel (f_water)
    f_water_model_type = trial.suggest_categorical(
        "f_water_model_type", ["linear", "lightgbm"]
    )  # Add "nn" later if needed

    f_water_model_params = {"model_type": f_water_model_type}
    if f_water_model_type == "linear":
        # No specific hyperparameters for simple LinearRegression
        pass
    elif f_water_model_type == "lightgbm":
        f_water_model_params["n_estimators"] = trial.suggest_int(
            "lgbm_n_estimators", 50, 500
        )
        f_water_model_params["learning_rate"] = trial.suggest_float(
            "lgbm_learning_rate", 0.01, 0.3, log=True
        )
        f_water_model_params["max_depth"] = trial.suggest_int("lgbm_max_depth", 3, 10)
        f_water_model_params["num_leaves"] = trial.suggest_int(
            "lgbm_num_leaves", 20, 100
        )
        f_water_model_params["reg_alpha"] = trial.suggest_float(
            "lgbm_reg_alpha", 1e-8, 1.0, log=True
        )
        f_water_model_params["reg_lambda"] = trial.suggest_float(
            "lgbm_reg_lambda", 1e-8, 1.0, log=True
        )

    # Weights for the loss components
    w_meta = trial.suggest_float("w_meta", 1e-5, 1.0, log=True)
    w_accuracy = trial.suggest_float("w_accuracy", 1e-5, 1.0, log=True)
    w_water = trial.suggest_float("w_water", 1e-5, 1.0, log=True)

    # 3. Instantiate and Run the WeightModel
    model = WeightModel(
        K_cal_kg=K_cal_kg,
        initial_M_base=initial_M_base,
        alpha=alpha,  # Pass alpha to the model
        f_water_model_params=f_water_model_params,
        look_back_window=look_back_window,
    )

    # Prepare data for f_water model training
    # We need to simulate the model once to get the Observed_Difference and other features
    # that f_water_model will be trained on.

    # This is a bit tricky: f_water_model needs to be trained *before* the main model run,
    # but its target (proportion 'p') depends on the main model's output.
    # A common approach is to train f_water_model on a "rolling" basis or in a separate loop
    # if the model is truly adaptive.

    # For Optuna, we will simplify:
    # 1. Run the WeightModel once with a dummy f_water_model (e.g., constant 0.5 for 'p').
    # 2. Collect the features (X_f_water) and targets (p_t) that the f_water_model *should* have predicted.
    #    The target 'p_t' can be derived from the actual W_obs, W_act_predicted_from_calories, and W_act_t.
    #    p_t = WR_t / Observed_Difference_t (if Observed_Difference_t != 0)
    # 3. Train the f_water_model on this collected data.
    # 4. Re-run the WeightModel with the *trained* f_water_model.
    # 5. Calculate losses.

    # Step 1 & 2: Simulate to collect f_water training data
    temp_model = WeightModel(
        K_cal_kg=K_cal_kg,
        initial_M_base=initial_M_base,
        alpha=alpha,  # Pass alpha to the temporary model as well
        f_water_model_params={
            "model_type": "linear"
        },  # Use a simple model for initial pass
        look_back_window=look_back_window,
    )
    temp_results = temp_model.run(data_df)

    X_f_water_train = []
    y_f_water_train = []

    nutritional_cols = [col for col in data_df.columns if col not in ["Pds", "Sport"]]

    for i in range(look_back_window, len(data_df)):
        Observed_Difference_t = temp_results["W_obs"].iloc[i] - (
            temp_results["W_act"].iloc[i - 1]
            + (
                temp_results["C_in"].iloc[i]
                - (
                    temp_results["M_base"].iloc[i - 1]
                    + temp_results["C_exp_activity"].iloc[i]
                )
            )
            / K_cal_kg
        )

        # Calculate the 'true' p_t from the temp_results
        # p_t = WR_t / Observed_Difference_t
        # If Observed_Difference_t is near zero, p_t can be unstable. Handle this.
        if abs(Observed_Difference_t) > 1e-6:  # Avoid division by zero
            true_p_t = temp_results["WR"].iloc[i] / Observed_Difference_t
            true_p_t = np.clip(true_p_t, 0.0, 1.0)  # Ensure p is within bounds
        else:
            true_p_t = 0.5  # Default if no significant difference

        # Prepare features for f_water_model training
        historical_features = model._prepare_f_water_features(i, data_df)
        if historical_features is None:  # Should not happen if i >= look_back_window
            continue

        current_nutritional_vector_t = data_df.iloc[i][nutritional_cols].values.tolist()

        X_f_water_list = (
            [Observed_Difference_t]
            + current_nutritional_vector_t
            + historical_features["historical_nutritional"].tolist()
            + historical_features["historical_pds"].tolist()
        )

        X_f_water_train.append(X_f_water_list)
        y_f_water_train.append(true_p_t)

    if not X_f_water_train:  # Handle case where data is too short for look_back_window
        return float("inf")  # Return a very high loss

    X_f_water_train = pd.DataFrame(X_f_water_train)
    y_f_water_train = pd.Series(y_f_water_train)

    # Step 3: Train the f_water_model
    model.f_water_model.fit(X_f_water_train, y_f_water_train)

    # Step 4: Re-run the WeightModel with the trained f_water_model
    results = model.run(data_df)

    # 5. Calculate Losses
    # L_meta (Metabolism Stability)
    M_base_series = results["M_base"]
    L_meta = np.sum(np.diff(M_base_series) ** 2)  # Sum of squared differences

    # L_accuracy (Prediction Accuracy)
    # Use a smoothed version of W_obs for L_accuracy
    # A simple rolling mean can serve as 'smoothed(W_obs(t))'
    smoothed_W_obs = (
        data_df["Pds"]
        .rolling(window=look_back_window, min_periods=1, center=True)
        .mean()
    )
    # Ensure alignment of indices
    common_index = results["W_act"].index.intersection(smoothed_W_obs.index)
    L_accuracy = mean_squared_error(
        results["W_act"].loc[common_index], smoothed_W_obs.loc[common_index]
    )

    # L_water (Water Retention Minimization)
    L_water = np.sum(results["WR"] ** 2)

    # Combined Objective
    total_loss = w_meta * L_meta + w_accuracy * L_accuracy + w_water * L_water

    return total_loss


if __name__ == "__main__":
    print("Running dummy example for Optuna Objective...")

    # Create dummy data_df similar to what data_processor.py would output
    dummy_data = {
        "Date": pd.to_datetime(
            [
                "2024-07-01",
                "2024-07-02",
                "2024-07-03",
                "2024-07-04",
                "2024-07-05",
                "2024-07-06",
                "2024-07-07",
                "2024-07-08",
                "2024-07-09",
                "2024-07-10",
                "2024-07-11",
                "2024-07-12",
                "2024-07-13",
                "2024-07-14",
            ]
        ),
        "Pds": [
            70.0,
            70.5,
            70.2,
            70.8,
            70.3,
            70.6,
            70.1,
            70.4,
            70.0,
            70.5,
            70.2,
            70.7,
            70.3,
            70.6,
        ],
        "sport": [
            300,
            200,
            0,
            400,
            100,
            0,
            250,
            150,
            0,
            300,
            200,
            0,
            400,
            100,
        ],  # Activity calories
        "calories": [
            2000,
            2200,
            1800,
            2500,
            1900,
            2100,
            2300,
            2000,
            1700,
            2400,
            2100,
            1900,
            2600,
            2000,
        ],
        "proteines": [100, 110, 90, 120, 95, 105, 115, 100, 85, 120, 100, 90, 130, 100],
        "glucides": [
            200,
            250,
            180,
            300,
            190,
            220,
            260,
            200,
            170,
            280,
            210,
            190,
            310,
            200,
        ],
        "lipides": [70, 80, 60, 90, 65, 75, 85, 70, 55, 80, 70, 60, 95, 70],
        "alcool": [0, 100, 0, 0, 50, 0, 0, 75, 0, 0, 0, 100, 0, 0],
    }
    dummy_df = pd.DataFrame(dummy_data).set_index("Date")

    # To run the objective function, we need a dummy trial.
    # Optuna's create_study().optimize() handles this.
    # For a quick test, we can mock a trial object.
    class MockTrial:
        def suggest_float(self, name, low, high, log=False):
            if name == "K_cal_kg":
                return 7700.0
            if name == "initial_M_base":
                return 2000.0
            if name == "alpha":
                return 0.1  # Dummy alpha for example
            if name == "w_meta":
                return 0.1
            if name == "w_accuracy":
                return 1.0
            if name == "w_water":
                return 0.01
            if name == "lgbm_learning_rate":
                return 0.05
            if name == "lgbm_reg_alpha":
                return 1e-6
            if name == "lgbm_reg_lambda":
                return 1e-6
            return (low + high) / 2  # Default for other floats

        def suggest_int(self, name, low, high):
            if name == "look_back_window":
                return 7
            if name == "lgbm_n_estimators":
                return 100
            if name == "lgbm_max_depth":
                return 5
            if name == "lgbm_num_leaves":
                return 31
            return (low + high) // 2

        def suggest_categorical(self, name, choices):
            if name == "f_water_model_type":
                return "linear"  # Test linear first
            return choices[0]

    mock_trial = MockTrial()

    try:
        loss = objective(mock_trial, dummy_df)
        print(f"\nDummy Objective Loss: {loss}")
    except Exception as e:
        print(f"\nError running dummy objective: {e}")
