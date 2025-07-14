# Part 3: Objective Function

**Goal:** Create the Optuna objective function that will be used to train the model.

**Files to Create/Modify:**
*   `objective.py`
*   `tests/test_objective.py`

**Implementation Details (`objective.py`):**

*   **Error Handling:** The script will not contain any `try-except` blocks. Any failure will cause the script to fail immediately.

*   **Core Model Execution (`run_model_simulation`):**
    *   **Signature:** `run_model_simulation(params: dict, features_df: pd.DataFrame) -> pd.DataFrame`
    *   **Purpose:** This function is the single source of truth for model execution. It takes parameters and data, runs the full `WeightModel` simulation, and returns a comprehensive DataFrame with all model outputs (`W_pred`, `W_act`, `M_base`, `WR`, etc.).
    *   **Reusability:** This function will be imported and used directly by the main training script (`train_model.py`) and the analysis script (`analyze_results.py`) to generate model predictions based on a given set of parameters.

*   **Loss Calculation Function (`calculate_loss`):**
    *   **Signature:** `calculate_loss(results_df: pd.DataFrame, w_meta: float, w_water: float) -> float`
    *   **Purpose:** This function encapsulates the logic for calculating the objective value. It takes the output from `run_model_simulation` and the fixed loss weights to compute the final loss.
    *   **Loss Components:** The total loss is a weighted sum of two opposing penalties:
        *   `L_meta`: `results_df['M_base'].diff().pow(2).mean()`
            *   **Purpose:** This penalizes large, rapid fluctuations in the base metabolism (`M_base`). It enforces the underlying assumption that metabolism should be a slowly varying component, preventing the model from making unrealistic, day-to-day jumps in metabolic rate to fit the data.
        *   `L_water`: `results_df['WR'].pow(2).mean()`
            *   **Purpose:** This acts as a regularization term, penalizing the model for relying too heavily on the water retention component (`WR`). By minimizing the magnitude of `WR`, it encourages the model to first explain weight changes through the more fundamental energy balance (`W_act`) before resorting to water fluctuations.
    *   **Total Loss:** `total_loss = (w_meta * L_meta) + (w_water * L_water)`
    *   It returns the final `total_loss` as a float.

*   **Optuna Objective Function (`objective`):**
    *   **Signature:** `objective(trial: optuna.Trial, features_df: pd.DataFrame, w_meta: float, w_water: float) -> float`
    *   **Purpose:** This function acts as the wrapper that connects Optuna to the model and loss functions. It is designed to be called by `study.optimize` and will receive the fixed loss weights.
    *   **Hyperparameter Suggestion (Optuna):**
        *   It will define the search space for the model's hyperparameters (e.g., `initial_M_base`, `alpha`, `look_back_window`) using `trial.suggest_*` methods. The loss weights (`w_meta`, `w_water`) are *not* tuned here; they are treated as fixed inputs.
    *   **Execution Flow:**
        1.  It will create the `params` dictionary for the model simulation, including the fixed `K_cal_kg` value of `7700`.
        2.  It will call `run_model_simulation(params, features_df)` to get the `results_df`.
        3.  It will call `calculate_loss(results_df, w_meta, w_water)` to get the `total_loss`.
    *   **Return Value:** It returns the `total_loss` to the Optuna study for optimization.

**Testing (`tests/test_objective.py`):**
*   Create a test to ensure that the `objective` function returns a float value.
*   Test the two-pass training logic to ensure that the `WaterRetentionModel` is correctly trained in the second pass.

**Definition of Done:**
*   The `objective.py` script is implemented as specified.
*   The `tests/test_objective.py` script exists and all tests pass.