import optuna
import pandas as pd
from data_processor import load_and_process_data
from objective import objective
import os


def tune_model(
    n_trials=100,
    study_name="weight_model_optimization",
    storage_path="sqlite:///db.sqlite3",
):
    """
    Runs the Optuna study to tune the WeightModel.

    Args:
        n_trials (int): Number of trials for the Optuna study.
        study_name (str): Name of the Optuna study.
        storage_path (str): Path for the Optuna study database (e.g., "sqlite:///db.sqlite3").
    """
    print("Starting model tuning process...")

    # 1. Load and preprocess data
    try:
        data_df = load_and_process_data()
        print(f"Data loaded and processed. Shape: {data_df.shape}")
        if data_df.empty:
            print("Error: Processed data is empty. Cannot proceed with tuning.")
            return
    except FileNotFoundError as e:
        print(
            f"Error: {e}. Please ensure 'processed_journal.csv' and 'variables.csv' are in the 'data/' directory."
        )
        print("You might need to run 'python3 process_nutrition_journal.py' first.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return

    # 2. Create or load Optuna study
    # Ensure the directory for the SQLite database exists
    db_dir = os.path.dirname(storage_path.replace("sqlite:///", ""))
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
        print(f"Created directory for Optuna storage: {db_dir}")

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
    )
    print(f"Optuna study '{study_name}' created/loaded.")

    # 3. Run optimization
    print(f"Running {n_trials} trials...")
    # Pass data_df as a fixed argument to the objective function
    study.optimize(lambda trial: objective(trial, data_df), n_trials=n_trials)

    print("\nOptimization finished.")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.value}")
    print("Best parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # Example usage:
    # To run the tuning, ensure you have Optuna installed: pip install optuna
    # And LightGBM if you plan to use it for f_water_model_type: pip install lightgbm

    # For a quick test, you can reduce n_trials
    tune_model(n_trials=5)
