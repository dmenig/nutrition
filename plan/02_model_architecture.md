# Part 2: Model Architecture

**Goal:** Create the core `WeightModel` and `WaterRetentionModel` classes.

**Files to Create/Modify:**
*   `weight_model.py`
*   `tests/test_weight_model.py`

**Implementation Details (`weight_model.py`):**

*   **Error Handling:** The script will not contain any `try-except` blocks. All operations are expected to succeed, and any failure (e.g., from missing data or incorrect calculations) will cause the script to fail immediately, ensuring that no results are produced from a partially failed run.

*   **`WaterRetentionModel` Class:**
    *   A wrapper around a regression model (e.g., `sklearn.linear_model.LinearRegression`) and a `sklearn.preprocessing.StandardScaler`.
    *   `__init__(self, model_type: str, **kwargs)`: Initializes the scaler and the specified regression model.
    *   `fit(self, X: pd.DataFrame, y: pd.Series)`: Scales the features `X` and fits the model.
    *   `predict(self, X: pd.DataFrame) -> np.ndarray`: Scales the input features `X` using the already-fitted scaler and returns the model's predictions.

*   **`WeightModel` Class:**
    *   This class will perform the main simulation. It will iterate through the `features.csv` data day by day, calculating the predicted weight based on the previous day's state and the current day's data.
    *   **`__init__(self, K_cal_kg: float, initial_M_base: float, alpha: float, f_water_model_params: dict, look_back_window: int)`**:
        *   `K_cal_kg`: Conversion factor from kcal delta to kg change.
        *   `initial_M_base`: The starting base metabolic rate.
        *   `alpha`: The EMA smoothing factor for updating metabolism.
        *   `f_water_model_params`: Dictionary of parameters for the `WaterRetentionModel`.
        *   `look_back_window`: The number of past days to use for creating features for the water retention model.
    *   **`_prepare_f_water_features(self, current_idx: int, data_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame`**:
        *   This method will construct the feature set for the `WaterRetentionModel` at a given time step (`current_idx`).
        *   **Input Features (from `features.csv` via `data_df`):**
            *   `carbs`, `sugar`, `sel`, `alcool`, `water`, `sport`
        *   **Engineered Features (from `results_df`):**
            *   Rolling averages of the input features over the `look_back_window`.
            *   Rolling standard deviations of the input features.
            *   `C_in_t - C_exp_t`: The calorie delta for the current day.
            *   `W_act_t-1`: The actual weight from the previous day.
    *   **`run(self, data_df: pd.DataFrame) -> pd.DataFrame`**:
        *   The core simulation loop that iterates from the first to the last day in `data_df`.
        *   For each day `t`, it calculates the day's values based on the state of day `t-1`.
        *   **Key Formulas:**
            *   `C_in_t = data_df.loc[t, 'calories']`
            *   `C_exp_t = M_t + data_df.loc[t, 'sport']`
            *   `W_act_t = W_act_t-1 + (C_in_t - C_exp_t) / K_cal_kg`
            *   `M_t = alpha * (C_in_t - C_exp_t - (W_obs_t - W_obs_t-1) * K_cal_kg) + (1 - alpha) * M_t-1` (Metabolism updated via EMA)
            *   `WR_t = f_water.predict(X_t)` (Water retention prediction)
            *   `W_pred_t = W_act_t + WR_t` (Final predicted weight)
        *   It will return a DataFrame (`results_df`) containing the full simulation results, including all intermediate calculations for each day.

**Testing (`tests/test_weight_model.py`):**
*   `test_water_retention_model`: Test the `fit` and `predict` methods.
*   `test_prepare_f_water_features`: Test the feature preparation logic.
*   `test_weight_model_run`: Test the `run` method with a small, controlled dataset and assert that the outputs are as expected.

**Definition of Done:**
*   The `weight_model.py` script is implemented as specified.
*   The `tests/test_weight_model.py` script exists and all tests pass.