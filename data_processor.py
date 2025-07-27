import pandas as pd
import numpy as np
import os
import re
from nutrition_calculator import (
    calculate_nutrient_from_formula,
    get_nutrient_context,
    calculate_nutrient_from_formula_with_context,
)
from utils import normalize_food_names


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

    # Create a mapping from original to normalized food names
    food_name_mapping = {
        row["Nom"]: normalize_food_names(row["Nom"])
        for _, row in variables_df.iterrows()
    }

    # Identify all nutrients available in the variables file (excluding 'Nom')
    all_nutrients = [
        col
        for col in variables_df.columns
        if col != "Nom" and not str(col).startswith("Unnamed")
    ]

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

        # Normalize food names within the formula
        for original_name, normalized_name in food_name_mapping.items():
            # Use regex to replace whole words only, ignoring case
            food_formula = re.sub(
                r"\b" + re.escape(original_name) + r"\b",
                normalized_name,
                food_formula,
                flags=re.IGNORECASE,
            )

        for nutrient in all_nutrients:
            # Use the pre-loaded context for calculation
            if food_formula:
                calculated_value = calculate_nutrient_from_formula_with_context(
                    food_formula, nutrient_contexts[nutrient], nutrient
                )
            else:
                calculated_value = 0
            daily_data[nutrient] = calculated_value
        daily_nutrients_data.append(daily_data)

    features_df = pd.DataFrame(daily_nutrients_data)
    features_df.set_index("Date", inplace=True)
    features_df.sort_index(inplace=True)

    return features_df
