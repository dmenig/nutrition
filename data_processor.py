import pandas as pd
import os
from nutrition_calculator import (
    calculate_nutrient_from_formula,
    get_nutrient_context,
    calculate_nutrient_from_formula_with_context,
)
from utils import strip_accents


def load_and_process_data(
    journal_path="data/processed_journal.csv", variables_path="data/variables.csv"
):
    """
    Loads the processed journal and variables data, then calculates daily nutritional intake
    for all available nutrients.

    Args:
        journal_path (str): Path to the processed journal CSV file.
        variables_path (str): Path to the variables CSV file.

    Returns:
        pd.DataFrame: A DataFrame with 'Date', 'Pds', 'Sport', and daily nutritional columns.
    """
    if not os.path.exists(journal_path):
        raise FileNotFoundError(f"Processed journal file not found: {journal_path}")
    if not os.path.exists(variables_path):
        raise FileNotFoundError(f"Variables file not found: {variables_path}")

    journal_df = pd.read_csv(journal_path)
    variables_df = pd.read_csv(variables_path)

    journal_df["Date"] = pd.to_datetime(journal_df["Date"])
    journal_df.set_index("Date", inplace=True)
    journal_df.sort_index(inplace=True)

    # Identify all nutrients available in the variables file (excluding 'Nom')
    all_nutrients = [col for col in variables_df.columns if col != "Nom"]

    print(f"Identified nutrients: {all_nutrients}")

    # Prepare a dictionary to store daily nutrient calculations
    daily_nutrients_data = []

    # Pre-load all nutrient contexts to avoid re-reading variables.csv in loop
    nutrient_contexts = {
        nutrient: get_nutrient_context(nutrient, variables_df)
        for nutrient in all_nutrients
    }

    for date, row in journal_df.iterrows():
        daily_data = {"Date": date, "Pds": row["Pds"], "Sport": row["Sport"]}
        food_formula = str(row["Nourriture"]) if pd.notna(row["Nourriture"]) else ""

        for nutrient in all_nutrients:
            # Use the pre-loaded context for calculation
            calculated_value = calculate_nutrient_from_formula_with_context(
                food_formula, nutrient_contexts[nutrient], nutrient
            )
            daily_data[nutrient] = calculated_value
        daily_nutrients_data.append(daily_data)

    features_df = pd.DataFrame(daily_nutrients_data)
    features_df.set_index("Date", inplace=True)
    features_df.sort_index(inplace=True)

    return features_df


if __name__ == "__main__":
    # Example usage:
    # Ensure process_nutrition_journal.py has been run to create processed_journal.csv
    # and variables.csv in the data/ directory.
    print("Running data processing...")
    try:
        # This part is for demonstration. In a real scenario, ensure these files exist.
        # You might need to run process_nutrition_journal.py first.
        # subprocess.run(["python3", "process_nutrition_journal.py"], check=True)

        # For testing, let's assume the files are already there or create dummy ones
        # if they don't exist for a quick run.
        # In a full run, the user would ensure these are generated.

        # Dummy data creation for testing if files don't exist
        if not os.path.exists("data/processed_journal.csv"):
            print("Creating dummy processed_journal.csv for demonstration.")
            dummy_journal = pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2024-07-01", "2024-07-02", "2024-07-03"]),
                    "Pds": [70.0, 70.5, 70.2],
                    "Nourriture": [
                        "100 * pain + 50 * fromage",
                        "200 * poulet",
                        "150 * riz",
                    ],
                    "Sport": ["", "", ""],
                }
            )
            os.makedirs("data", exist_ok=True)
            dummy_journal.to_csv("data/processed_journal.csv", index=False)

        if not os.path.exists("data/variables.csv"):
            print("Creating dummy variables.csv for demonstration.")
            dummy_variables = pd.DataFrame(
                {
                    "Nom": ["pain", "fromage", "poulet", "riz"],
                    "Calories / 100g": [250, 400, 165, 130],
                    "Proteines / 100g": [10, 25, 30, 3],
                    "Glucides / 100g": [50, 2, 0, 28],
                    "Lipides / 100g": [2, 30, 5, 0],
                    "Alcool / 100g": [0, 0, 0, 0],
                }
            )
            dummy_variables.to_csv("data/variables.csv", index=False)

        features_df = load_and_process_data()
        print("\nProcessed Features DataFrame (first 3 rows):")
        print(features_df.head(3))
        print(f"\nDataFrame shape: {features_df.shape}")
        print(f"DataFrame columns: {features_df.columns.tolist()}")

    except FileNotFoundError as e:
        print(
            f"Error: {e}. Please ensure 'processed_journal.csv' and 'variables.csv' are in the 'data/' directory."
        )
        print("You might need to run 'python3 process_nutrition_journal.py' first.")
    except Exception as e:
        print(f"An error occurred during data processing: {e}")


def save_normalized_variables(
    variables_path="data/variables.csv",
    normalized_path="data/normalized_variables.csv",
):
    """
    Reads the variables CSV, normalizes the 'Nom' column to be valid Python
    identifiers, and saves the result to a new CSV file.

    Normalization includes:
    - Stripping accents.
    - Converting to lowercase.
    - Replacing spaces and special characters with underscores.
    """
    if not os.path.exists(variables_path):
        raise FileNotFoundError(f"Variables file not found: {variables_path}")

    variables_df = pd.read_csv(variables_path)

    # Normalize the 'Nom' column
    variables_df["Nom"] = (
        variables_df["Nom"]
        .apply(strip_accents)
        .str.lower()
        .str.replace(r"[^a-zA-Z0-9_]+", "_", regex=True)
    )

    # Save the normalized DataFrame
    variables_df.to_csv(normalized_path, index=False)
    print(f"Normalized variables saved to {normalized_path}")
