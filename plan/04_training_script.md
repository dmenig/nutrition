# Part 4: Training Script

**Goal:** Create the entry point for running the Optuna study.

**Files to Create/Modify:**
*   `train_model.py`
*   `tests/test_training.py`

**Implementation Details (`train_model.py`):**
*   **Purpose:** The main entry point for finding the optimal loss weights and then running the final hyperparameter optimization study.
*   **Error Handling & Logging:** The script will not use `try-except` blocks and will only show Optuna's trial results.

*   **Phase 1: Loss Weight Grid Search:**
    1.  **Define Search Space:** The script will define a grid of potential values for `w_meta` and `w_water`.
    2.  **Iterate and Evaluate:** It will loop through each `(w_meta, w_water)` pair in the grid.
        *   For each pair, it will run a full Optuna study (`study.optimize` with `n_trials=200`, `n_startup_trials=50`) to find the best model parameters for that specific loss definition.
        *   It will then evaluate the resulting best model against a set of sanity checks.
    3.  **Sanity Checks:**
        *   The evaluation will reuse the existing `run_model_simulation` function. No new evaluation functions will be created for this step.
        *   `Metabolism Range`: Check if `M_base` in the returned DataFrame stays strictly between 1400 and 3200.
        *   `Weight Tracking`: Check if the `W_act` (actual weight) provides a reasonable, smoothed version of the observed weight `pds`.
    4.  **Select Best Weights:** The `(w_meta, w_water)` pair that produces the model that best satisfies the sanity checks will be selected as the optimal loss definition.

*   **Phase 2: Final Training Run:**
    1.  **Create Final Study:** It will create a new, final Optuna study.
    2.  **Run Final Optimization:** It will call `study.optimize` one last time, using the best `w_meta` and `w_water` found in Phase 1.
    3.  **Save Artifacts:**
        *   It will save the final best model parameters to `best_params.json`.
        *   It will save the optimal loss weights to a new file, `loss_weights.json`.
        *   It will generate and save the final model simulation results to `data/final_results.csv`.

**Testing (`tests/test_training.py`):**
*   Create a test to ensure that `train_model.py` runs without errors.
*   Assert that a `best_params.json` file is created after the script runs.

**Definition of Done:**
*   The `train_model.py` script is implemented as specified.
*   The `tests/test_training.py` script exists and all tests pass.