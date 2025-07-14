import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from weight_model import WeightModel
from data_processor import load_and_process_data
from water_retention_model import WaterRetentionModel


def run_and_analyze_model(best_params, data_df):
    """
    Runs the WeightModel with the best parameters and performs analysis.

    Args:
        best_params (dict): Dictionary of best parameters obtained from Optuna.
        data_df (pd.DataFrame): The preprocessed DataFrame containing all features.

    Returns:
        dict: Results from the model run.
    """
    print("\n--- Running Model with Best Parameters ---")
    K_cal_kg = best_params["K_cal_kg"]
    initial_M_base = best_params["initial_M_base"]
    alpha = best_params["alpha"]  # New: alpha for EMA smoothing
    look_back_window = best_params["look_back_window"]
    f_water_model_params = {"model_type": best_params["f_water_model_type"]}
    if f_water_model_params["model_type"] == "lightgbm":
        f_water_model_params["n_estimators"] = best_params["lgbm_n_estimators"]
        f_water_model_params["learning_rate"] = best_params["lgbm_learning_rate"]
        f_water_model_params["max_depth"] = best_params["lgbm_max_depth"]
        f_water_model_params["num_leaves"] = best_params["lgbm_num_leaves"]
        f_water_model_params["reg_alpha"] = best_params["lgbm_reg_alpha"]
        f_water_model_params["reg_lambda"] = best_params["lgbm_reg_lambda"]

    model = WeightModel(
        K_cal_kg=K_cal_kg,
        initial_M_base=initial_M_base,
        alpha=alpha,  # Pass alpha to the model
        f_water_model_params=f_water_model_params,
        look_back_window=look_back_window,
    )

    # Train f_water_model before running the main model
    # This part is duplicated from objective.py, but necessary for standalone analysis
    X_f_water_train = []
    y_f_water_train = []
    nutritional_cols = [col for col in data_df.columns if col not in ["Pds", "Sport"]]

    # Create a temporary model instance to generate the 'true' p_t for f_water training
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

        if abs(Observed_Difference_t) > 1e-6:
            true_p_t = temp_results["WR"].iloc[i] / Observed_Difference_t
            true_p_t = np.clip(true_p_t, 0.0, 1.0)
        else:
            true_p_t = 0.5

        historical_features = model._prepare_f_water_features(i, data_df)
        if historical_features is None:
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

    if not X_f_water_train:
        print("Not enough data to train f_water model. Skipping analysis.")
        return None

    X_f_water_train = pd.DataFrame(X_f_water_train)
    y_f_water_train = pd.Series(y_f_water_train)
    model.f_water_model.fit(X_f_water_train, y_f_water_train)

    results = model.run(data_df)
    print("Model run complete.")
    return results


def sanity_check(results, K_cal_kg):
    """
    Performs the sanity check on the model results.

    Args:
        results (dict): Dictionary of model results.
        K_cal_kg (float): Energy equivalent of 1kg of body weight.
    """
    print("\n--- Sanity Check: Long-term Energy Conservation ---")
    W_act_initial = results["W_act"].iloc[0]
    W_act_final = results["W_act"].iloc[-1]

    C_in_total = results["C_in"].sum()
    C_exp_activity_total = results["C_exp_activity"].sum()

    # Estimate total M_base over the period (average M_base * num_days)
    # Or, sum of daily M_base values
    M_base_total_exp = results["M_base"].sum()

    total_C_delta = C_in_total - (M_base_total_exp + C_exp_activity_total)

    predicted_weight_change_from_calories = total_C_delta / K_cal_kg
    actual_weight_change = W_act_final - W_act_initial

    print(f"Actual W_act Change: {actual_weight_change:.2f} kg")
    print(
        f"Predicted W_act Change from Total Calories: {predicted_weight_change_from_calories:.2f} kg"
    )
    print(
        f"Discrepancy: {abs(actual_weight_change - predicted_weight_change_from_calories):.2f} kg"
    )

    if (
        abs(actual_weight_change - predicted_weight_change_from_calories) < 1.0
    ):  # Threshold for "close"
        print(
            "Sanity Check PASSED: Long-term actual weight change is consistent with total calorie balance."
        )
    else:
        print(
            "Sanity Check FAILED: Significant discrepancy between actual weight change and total calorie balance."
        )


def plot_results(results):
    """
    Plots the model results: observed vs. actual weight, water retention, and metabolism.

    Args:
        results (dict): Dictionary of model results.
    """
    if results is None:
        print("No results to plot.")
        return

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle("Weight Model Analysis", fontsize=16)

    # Plot 1: Observed vs. Actual Weight
    axes[0].plot(
        results["W_obs"].index,
        results["W_obs"],
        label="Observed Weight (Pds)",
        marker="o",
        linestyle="-",
        markersize=4,
        alpha=0.7,
    )
    axes[0].plot(
        results["W_act"].index,
        results["W_act"],
        label="Actual Weight (W_act)",
        marker="x",
        linestyle="--",
        markersize=4,
        alpha=0.8,
    )
    axes[0].set_ylabel("Weight (kg)")
    axes[0].set_title("Observed vs. Actual Weight")
    axes[0].legend()

    # Plot 2: Water Retention
    axes[1].bar(
        results["WR"].index,
        results["WR"],
        width=0.8,
        label="Water Retention (WR)",
        color="skyblue",
        alpha=0.7,
    )
    axes[1].axhline(0, color="grey", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Water Retention (kg)")
    axes[1].set_title("Daily Water Retention")
    axes[1].legend()

    # Plot 3: Base Metabolism
    axes[2].plot(
        results["M_base"].index,
        results["M_base"],
        label="Base Metabolism (M_base)",
        color="salmon",
        marker="s",
        linestyle="-",
        markersize=4,
        alpha=0.7,
    )
    axes[2].set_ylabel("Metabolism (kcal/day)")
    axes[2].set_title("Estimated Base Metabolism Over Time")
    axes[2].legend()
    axes[2].set_xlabel("Date")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def plot_feature_importance(model_instance):
    """
    Plots feature importance for the f_water model if available.

    Args:
        model_instance (WeightModel): The trained WeightModel instance.
    """
    if (
        not hasattr(model_instance, "f_water_model")
        or model_instance.f_water_model is None
    ):
        print("No f_water model found in the WeightModel instance.")
        return

    importance = model_instance.f_water_model.get_feature_importance()
    if isinstance(importance, dict) and "Error" in importance:
        print(f"Cannot plot feature importance: {importance['Error']}")
        return
    if isinstance(importance, dict) and "Warning" in importance:
        print(f"Cannot plot feature importance: {importance['Warning']}")
        return

    if isinstance(importance, dict):
        features = list(importance.keys())
        values = list(importance.values())
    else:  # Assume it's a numpy array for models like LightGBM without explicit names
        features = [f"Feature {i}" for i in range(len(importance))]
        values = importance

    # Sort features by importance
    sorted_idx = np.argsort(values)[::-1]
    features = [features[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=values, y=features)
    plt.title("f_water Model Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Running dummy analysis example...")

    # This part would typically be run after Optuna has found the best parameters.
    # For demonstration, we'll use dummy data and dummy best_params.

    # Create dummy data_df (similar to data_processor.py output)
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

    # Dummy best parameters (from a hypothetical Optuna run)
    dummy_best_params = {
        "K_cal_kg": 7700.0,
        "initial_M_base": 2000.0,
        "alpha": 0.1,  # Dummy alpha for example
        "look_back_window": 7,
        "f_water_model_type": "linear",
        # Add LightGBM params if testing that type:
        # "lgbm_n_estimators": 100,
        # "lgbm_learning_rate": 0.05,
        # "lgbm_max_depth": 5,
        # "lgbm_num_leaves": 31,
        # "lgbm_reg_alpha": 1e-6,
        # "lgbm_reg_lambda": 1e-6,
        "w_meta": 0.1,
        "w_accuracy": 1.0,
        "w_water": 0.01,
    }

    # Run analysis
    results = run_and_analyze_model(dummy_best_params, dummy_df)
    if results is not None:
        sanity_check(results, dummy_best_params["K_cal_kg"])
        plot_results(results)

        # To plot feature importance, we need the actual model instance from run_and_analyze_model
        # This requires modifying run_and_analyze_model to return the model instance as well.
        # For now, we'll just pass a dummy model or skip this plot in the example.

        # Let's modify run_and_analyze_model to return the model instance too.
        # For this example, we'll create a dummy model instance for plotting.

        dummy_model_instance = WeightModel(
            K_cal_kg=dummy_best_params["K_cal_kg"],
            initial_M_base=dummy_best_params["initial_M_base"],
            alpha=dummy_best_params["alpha"],  # Pass alpha to dummy model
            f_water_model_params={"model_type": "linear"},  # Use linear for dummy plot
            look_back_window=dummy_best_params["look_back_window"],
        )
        # Fit the dummy f_water model for plotting purposes
        nutritional_cols_count = len(
            [col for col in dummy_df.columns if col not in ["Pds", "sport"]]
        )
        expected_f_water_features_count = (
            1
            + nutritional_cols_count
            + (nutritional_cols_count * dummy_best_params["look_back_window"])
            + dummy_best_params["look_back_window"]
        )
        X_f_water_train_dummy = np.random.rand(
            len(dummy_df) - dummy_best_params["look_back_window"],
            expected_f_water_features_count,
        )
        y_f_water_train_dummy = np.random.rand(
            len(dummy_df) - dummy_best_params["look_back_window"]
        )
        dummy_model_instance.f_water_model.fit(
            pd.DataFrame(
                X_f_water_train_dummy,
                columns=[
                    f"feature_{i}" for i in range(expected_f_water_features_count)
                ],
            ),
            y_f_water_train_dummy,
        )

        plot_feature_importance(dummy_model_instance)
