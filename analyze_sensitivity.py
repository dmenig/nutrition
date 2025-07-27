import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# Import the model class from the training script
from train_model import FinalModel

def run_simulation(model, nutrition_data_tensor, device):
    """Runs the model on the given data and returns the metabolism prediction."""
    model.eval()
    with torch.no_grad():
        base_metabolisms = model(nutrition_data_tensor.to(device))
    return base_metabolisms.squeeze().cpu().numpy()

def main():
    # --- Load Data and Model ---
    try:
        # Use final_results.csv as the source of truth for the data the model was trained on
        features_df = pd.read_csv('data/final_results.csv')
        with open('best_params.json', 'r') as f:
            params = json.load(f)
            normalization_stats = params['normalization']
    except FileNotFoundError:
        print("Error: Required data files not found.")
        print("Please run train_model.py first to generate the model and data.")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    nutrition_cols = ["calories", "carbs", "sugar", "sel", "alcool", "water"]
    nutrition_input_size = len(nutrition_cols)
    
    # We need the initial weight guess to initialize the model, even if it's not used in this script
    # We'll use the overall mean as a placeholder
    initial_weight_guess = 0.0 # Placeholder
    
    model = FinalModel(nutrition_input_size, initial_weight_guess)
    model.load_state_dict(torch.load("recurrent_model.pth", map_location=device))
    model.to(device)
    print("Model and data loaded successfully.")

    # --- Prepare Data for Simulation ---
    # The data from final_results.csv is already normalized, so we just need the tensor
    base_nutrition_tensor = torch.tensor(features_df[nutrition_cols].values, dtype=torch.float32).unsqueeze(0)

    # --- Define Scenarios ---
    scenarios = {
        "control": base_nutrition_tensor.clone(),
        "high_salt": base_nutrition_tensor.clone(),
        "high_carbs": base_nutrition_tensor.clone(),
        "high_water": base_nutrition_tensor.clone(),
    }

    # Modify the last 30 days for each scenario
    simulation_days = 30
    salt_col_index = nutrition_cols.index("sel")
    carbs_col_index = nutrition_cols.index("carbs")
    water_col_index = nutrition_cols.index("water")

    # Increase by 2 standard deviations (since data is normalized, this is just adding 2)
    scenarios["high_salt"][0, -simulation_days:, salt_col_index] += 2.0
    scenarios["high_carbs"][0, -simulation_days:, carbs_col_index] += 2.0
    scenarios["high_water"][0, -simulation_days:, water_col_index] += 2.0
    print(f"Simulating {len(scenarios)} scenarios over the last {simulation_days} days...")

    # --- Run Simulations ---
    results = {}
    for name, scenario_tensor in scenarios.items():
        results[name] = run_simulation(model, scenario_tensor, device)
    print("Simulations complete.")

    # ---- DEBUG: Print summary statistics of the results ---
    print("\n--- Sensitivity Analysis Results Summary ---")
    for name, data in results.items():
        print(f"Scenario: {name}")
        print(f"  Shape: {data.shape}")
        # Un-normalize metabolism for interpretation by multiplying by 1000
        mean_metabolism = np.mean(data) * 1000
        std_metabolism = np.std(data) * 1000
        min_metabolism = np.min(data) * 1000
        max_metabolism = np.max(data) * 1000
        print(f"  Mean Metabolism: {mean_metabolism:.2f} kcal")
        print(f"  Std Dev: {std_metabolism:.2f} kcal")
        print(f"  Min Metabolism: {min_metabolism:.2f} kcal")
        print(f"  Max Metabolism: {max_metabolism:.2f} kcal")
        if np.isnan(data).any():
            print(f"  WARNING: Contains NaN values!")
    print("------------------------------------------\n")


    # --- Plot Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    time_index = np.arange(len(features_df))
    plot_range = slice(-simulation_days, None)

    ax.plot(time_index[plot_range], results["control"][plot_range] * 1000, label="Control (Original Data)", color='k', linestyle='--', linewidth=2)
    ax.plot(time_index[plot_range], results["high_salt"][plot_range] * 1000, label="High Salt Scenario", color='r', linewidth=2.5, alpha=0.8)
    ax.plot(time_index[plot_range], results["high_carbs"][plot_range] * 1000, label="High Carbs Scenario", color='b', linewidth=2.5, alpha=0.8)
    ax.plot(time_index[plot_range], results["high_water"][plot_range] * 1000, label="High Water Intake Scenario", color='g', linewidth=2.5, alpha=0.8)

    ax.axhline(0, color='gray', linestyle='-', alpha=0.7)
    ax.set_title("Sensitivity Analysis: How Nutrition Affects Predicted Base Metabolism", fontsize=16, pad=20)
    ax.set_xlabel("Days", fontsize=12)
    ax.set_ylabel("Predicted Base Metabolism (kcal/day)", fontsize=12)
    ax.legend(fontsize=11)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    fig.tight_layout()
    plt.savefig("sensitivity_analysis.png", dpi=300)
    plt.show()
    print("Plot saved to sensitivity_analysis.png")

if __name__ == "__main__":
    main() 