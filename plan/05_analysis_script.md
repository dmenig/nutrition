# Part 5: Analysis Script

**Goal:** Create a script to visualize the performance of the best model and perform a final sanity check.

**Files to Create/Modify:**
*   `analyze_results.py`
*   `tests/test_analysis.py`

**Implementation Details (`analyze_results.py`):**
*   **Purpose:** To load the final, optimized model artifacts, run the simulation once, and generate plots and a final sanity check. This script will not recompute or recalculate any values; it is for visualization and validation only.
*   **Logic:**
    1.  **Load Artifacts:**
        *   Load the `features_df` from `data/features.csv`.
        *   Load the `best_params.json` file.
        *   Load the `loss_weights.json` file.
    2.  **Run Final Simulation:**
        *   Call the `run_model_simulation` function from `objective.py` with the loaded best parameters and features. This is the only computation in the script and reuses the existing function.
    3.  **Generate Plots:**
        *   The script will generate and save several plots to a `plots/` directory using the DataFrame from the simulation.
        *   **Plot 1: Weight Overview:** Time series of observed weight (`pds`), actual weight (`W_act`), and predicted weight (`W_pred`).
        *   **Plot 2: Water Retention:** Time series of water retention (`WR`).
        *   **Plot 3: Metabolism:** Time series of base metabolism (`M_base`).
    4.  **Perform Sanity Check:**
        *   This is a critical final check to ensure the model is physically plausible.
        *   `total_calorie_delta = (features_df['calories'] - features_df['sport'] - results_df['M_base']).sum()`
        *   `total_weight_change_kg = results_df['W_act'].iloc[-1] - results_df['W_act'].iloc[0]`
        *   The script will assert that `abs(total_calorie_delta / 7700 - total_weight_change_kg) < 1e-9`. A failure here indicates a fundamental flaw in the model's energy balance calculation.

**Testing (`tests/test_analysis.py`):**
*   Create a test to ensure that `analyze_results.py` runs without errors.
*   Assert that the sanity check passes.

**Definition of Done:**
*   The `analyze_results.py` script is implemented as specified.
*   The `tests/test_analysis.py` script exists and all tests pass.