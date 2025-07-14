import os
import subprocess
import json
import pytest


def test_training_script_runs_and_creates_artifacts():
    """
    Tests that the training script runs without errors and creates the expected artifacts.
    """
    # Define the paths for the artifacts that should be created
    best_params_path = "best_params.json"
    loss_weights_path = "loss_weights.json"
    results_path = "data/final_results.csv"

    # Clean up any old artifacts before running the test
    if os.path.exists(best_params_path):
        os.remove(best_params_path)
    if os.path.exists(loss_weights_path):
        os.remove(loss_weights_path)
    if os.path.exists(results_path):
        os.remove(results_path)
    if os.path.exists("data") and not os.listdir("data"):
        os.rmdir("data")

    # Run the training script as a subprocess
    result = subprocess.run(
        ["python", "train_model.py"], capture_output=True, text=True
    )

    # Assert that the script ran successfully
    assert result.returncode == 0, (
        f"Training script failed with error:\n{result.stderr}"
    )

    # Assert that the expected artifacts were created
    assert os.path.exists(best_params_path), f"'{best_params_path}' was not created."
    assert os.path.exists(loss_weights_path), f"'{loss_weights_path}' was not created."
    assert os.path.exists(results_path), f"'{results_path}' was not created."

    # Optional: Clean up the created artifacts after the test
    os.remove(best_params_path)
    os.remove(loss_weights_path)
    os.remove(results_path)
    if os.path.exists("data") and not os.listdir("data"):
        os.rmdir("data")
