import json
import itertools
import pandas as pd
import numpy as np
import optuna
import os

from objective import objective, run_model_simulation
from build_features import build_features


def find_best_loss_weights(features_df, f_water_model_params):
    """
    Performs a grid search to find the best loss weights (w_meta, w_water).
    """
    w_meta_grid = [0.1, 1.0, 10.0]
    w_water_grid = [0.1, 1.0, 10.0]

    best_weights = None
    best_sanity_score = -np.inf

    search_space = list(itertools.product(w_meta_grid, w_water_grid))

    for w_meta, w_water in search_space:
        print(f"Testing weights: w_meta={w_meta}, w_water={w_water}")

        def objective_with_weights(trial):
            return objective(
                trial,
                features_df,
                w_meta=w_meta,
                w_water=w_water,
                f_water_model_params=f_water_model_params,
            )

        study = optuna.create_study(direction="minimize")
        study.optimize(objective_with_weights, n_trials=10)

        best_params = study.best_params
        best_params["K_cal_kg"] = 7700  # Add fixed value
        results_df = run_model_simulation(
            best_params, features_df, f_water_model_params
        )

        weight_tracking_score = -np.std(results_df["W_act"] - results_df["pds"])

        if weight_tracking_score > best_sanity_score:
            best_sanity_score = weight_tracking_score
            best_weights = {"w_meta": w_meta, "w_water": w_water}

    return best_weights


def final_training(loss_weights, features_df, f_water_model_params):
    """
    Runs the final hyperparameter optimization with the best loss weights.
    """
    print("Running final training with best weights:", loss_weights)

    w_meta = loss_weights["w_meta"]
    w_water = loss_weights["w_water"]

    def final_objective(trial):
        return objective(
            trial,
            features_df,
            w_meta=w_meta,
            w_water=w_water,
            f_water_model_params=f_water_model_params,
        )

    final_study = optuna.create_study(direction="minimize")
    final_study.optimize(final_objective, n_trials=10)

    best_params = final_study.best_params
    best_params["K_cal_kg"] = 7700  # Add fixed value
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    with open("loss_weights.json", "w") as f:
        json.dump(loss_weights, f, indent=4)

    if not os.path.exists("data"):
        os.makedirs("data")

    final_results_df = run_model_simulation(
        best_params, features_df, f_water_model_params
    )
    final_results_df.to_csv("data/final_results.csv", index=False)

    print("Final training complete. Artifacts saved.")


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = pd.read_csv("data/processed_journal.csv")
    features_df = build_features(df)
    if "calories_/_100g" in features_df.columns:
        features_df.rename(columns={"calories_/_100g": "calories"}, inplace=True)
    f_water_model_params = {}

    optimal_weights = find_best_loss_weights(features_df, f_water_model_params)

    if optimal_weights:
        final_training(optimal_weights, features_df, f_water_model_params)
    else:
        print("Could not find suitable loss weights that pass sanity checks.")


if __name__ == "__main__":
    main()
