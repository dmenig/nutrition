import json
import os
import pandas as pd
import matplotlib.pyplot as plt

from objective import run_model_simulation


def analyze_results():
    """
    Loads the final, optimized model artifacts, runs the simulation once,
    and generates plots and a final sanity check.
    """
    # Load Artifacts
    features_df = pd.read_csv("data/features.csv")
    with open("best_params.json", "r") as f:
        best_params = json.load(f)
    with open("loss_weights.json", "r") as f:
        loss_weights = json.load(f)

    # The training script uses an empty dict for this
    f_water_model_params = {}

    # Run Final Simulation
    results_df = run_model_simulation(best_params, features_df, f_water_model_params)

    # Create plots directory if it doesn't exist
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Generate Plots
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df["pds"], label="Observed Weight (pds)")
    plt.plot(results_df.index, results_df["W_act"], label="Actual Weight (W_act)")
    plt.plot(
        results_df.index, results_df["W_pred_t"], label="Predicted Weight (W_pred)"
    )
    plt.title("Weight Overview")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.legend()
    plt.savefig("plots/weight_overview.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df["WR_t"], label="Water Retention (WR)")
    plt.title("Water Retention")
    plt.xlabel("Date")
    plt.ylabel("Water Retention")
    plt.legend()
    plt.savefig("plots/water_retention.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df["M_base"], label="Base Metabolism (M_base)")
    plt.title("Metabolism")
    plt.xlabel("Date")
    plt.ylabel("Metabolism")
    plt.legend()
    plt.savefig("plots/metabolism.png")
    plt.close()

    # Perform Sanity Check
    total_calorie_delta = (
        features_df["calories"] - features_df["sport"] - results_df["M_base"]
    ).sum()
    total_weight_change_kg = results_df["W_act"].iloc[-1] - results_df["W_act"].iloc[0]

    # K_cal_kg is a fixed value of 7700
    K_cal_kg = 7700

    print(f"Total calorie delta: {total_calorie_delta}")
    print(f"Total weight change (kg): {total_weight_change_kg}")
    print(f"Calorie-equivalent weight change (kg): {total_calorie_delta / K_cal_kg}")

    assert abs(total_calorie_delta / K_cal_kg - total_weight_change_kg) < 0.1, (
        "Sanity check failed"
    )
    print("Sanity check passed.")


if __name__ == "__main__":
    analyze_results()
