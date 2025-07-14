# Nutrition Analysis

This project contains a collection of Python scripts for analyzing nutrition and sport data from a personal journal.

## Features

*   **Data Extraction**: Scripts to extract data from spreadsheet files.
*   **Data Processing**: Tools to process and clean journal entries for food, sport, and weight.
*   **Nutrition Calculation**: Functions to calculate nutritional information based on food entries.
*   **Analysis**: Scripts to analyze trends and provide insights into nutrition and sport activities.

## Scripts

-   `process_nutrition_journal.py`: Main script to process the nutrition journal, extracting daily nutrition and weight data.
-   `calculate_nutrition.py`: Contains functions for nutrition calculations based on food formulas.
-   `data_processor.py`: Loads processed journal and variables, then calculates a comprehensive daily nutritional feature set.
-   `water_retention_model.py`: Implements the `f_water` component, a flexible machine learning model for attributing weight differences to water retention.
-   `weight_model.py`: The core weight prediction model, which estimates actual weight, water retention, and adapts base metabolism.
-   `objective.py`: The Optuna objective function for tuning the `WeightModel` and `WaterRetentionModel` hyperparameters.
-   `tune_model.py`: The main script to run the Optuna optimization study.
-   `analysis.py`: Contains functions for sanity checks, plotting results (observed vs. actual weight, water retention, metabolism), and visualizing feature importance.
-   `model_plan.md`: Detailed planning document for the weight and metabolism model.

## How to Use the Weight Model

### 1. Data Preparation

Ensure your nutrition journal data is processed into `data/processed_journal.csv` and `data/variables.csv`.
If these files do not exist, run:
```bash
python3 process_nutrition_journal.py
```

### 2. Run the Optuna Tuning Study

To find the best parameters for the weight model, run the tuning script. This will save the study results in `db.sqlite3`.
```bash
python3 tune_model.py
```
You can adjust the number of trials by passing `n_trials` as an argument (e.g., `python3 tune_model.py --n_trials 500`).

### 3. Analyze the Best Model

After tuning, you can analyze the best model's performance and visualize its outputs.
You will need to retrieve the best parameters from the Optuna study (e.g., by inspecting `db.sqlite3` or running a small script to query it).

Example of how to get best params (can be added to a utility script or run interactively):
```python
import optuna
study = optuna.load_study(study_name="weight_model_optimization", storage="sqlite:///db.sqlite3")
best_params = study.best_trial.params
print(best_params)
```

Then, use these `best_params` with the `analysis.py` script.
```python
# Example usage within a Python script or interactive session
from data_processor import load_and_process_data
from analysis import run_and_analyze_model, sanity_check, plot_results, plot_feature_importance

# Load data
data_df = load_and_process_data()

# Replace with your actual best_params from Optuna
best_params = {
    "K_cal_kg": 7700.0,
    "initial_M_base": 2000.0,
    "look_back_window": 7,
    "f_water_model_type": "linear", # or "lightgbm"
    # Include lightgbm params if f_water_model_type is lightgbm
    # "lgbm_n_estimators": 100,
    # "lgbm_learning_rate": 0.05,
    # "lgbm_max_depth": 5,
    # "lgbm_num_leaves": 31,
    # "lgbm_reg_alpha": 1e-6,
    # "lgbm_reg_lambda": 1e-6,
    "w_meta": 0.1,
    "w_accuracy": 1.0,
    "w_water": 0.01
}

# Run analysis
results = run_and_analyze_model(best_params, data_df)
if results is not None:
    sanity_check(results, best_params["K_cal_kg"])
    plot_results(results)
    
    # To plot feature importance, you need to pass the model instance from run_and_analyze_model
    # run_and_analyze_model currently returns only results.
    # You might need to modify run_and_analyze_model to return the model instance as well,
    # or re-instantiate the model with best_params and fit it for plotting feature importance.
    
    # For now, if you want to see feature importance, you'd need to manually
    # instantiate and fit the model with the best parameters.
    # Example:
    # from weight_model import WeightModel
    # best_model_instance = WeightModel(
    #     K_cal_kg=best_params["K_cal_kg"],
    #     initial_M_base=best_params["initial_M_base"],
    #     f_water_model_params={"model_type": best_params["f_water_model_type"], ...}, # Pass all f_water params
    #     look_back_window=best_params["look_back_window"]
    # )
    # # You would need to re-train the f_water_model here on the full data
    # # using the same logic as in objective.py or analysis.py
    # # This is a bit redundant but necessary for standalone plotting.
    # plot_feature_importance(best_model_instance)
```