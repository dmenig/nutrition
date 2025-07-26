import json
import os
import pandas as pd
import matplotlib.pyplot as plt

from objective import run_model_simulation


def main():
    """
    Loads the final, optimized model artifacts, runs the simulation once,
    and generates plots and a final sanity check.
    """
    # Load Artifacts
    features_df = pd.read_csv("data/features.csv").fillna(0)
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


if __name__ == "__main__":
    main()
