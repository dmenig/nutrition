import data_processor
import sport_formulas
import nutrition_calculator
import pandas as pd
import ast
from utils import SafeSportFormulaEvaluator, normalize_food_names


def calculate_sport_calories(row: pd.Series) -> float:
    """
    Calculates calories burned from a sport formula string.

    Args:
        row: A pandas Series representing a row of the DataFrame.
             It must contain 'Sport', 'Pds', 'distance', and 'duration' columns.

    Returns:
        The calculated calories burned, or 0 if the formula is invalid.
    """
    sport_formula = row["Sport"]
    weight = row["Pds"]
    distance = row.get("distance", 0.0)
    duration = row.get("duration", 0.0)

    if not isinstance(sport_formula, str) or not sport_formula.strip():
        return 0.0

    sport_formula = sport_formula.strip()
    node = ast.parse(sport_formula, mode="eval")

    # Prepare the context for the formula evaluator
    # It needs the sport functions and the current weight, distance, and duration
    context = sport_formulas.SPORT_FUNCTIONS.copy()
    context["WEIGHT"] = weight
    context["distance_meters"] = distance
    context["duration_minutes"] = duration
    context["additional_weight_kg"] = 0  # Placeholder for now

    # Evaluate the formula
    evaluator = SafeSportFormulaEvaluator(context)
    return evaluator.visit(node.body)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds features from the raw data.
    """
    print("--- Starting Feature Building ---")

    # Specifically rename 'calories_/_100g' to 'calories'
    df.rename(columns={"Calories / 100g": "calories"}, inplace=True)

    # Calculate sport calories and handle original 'Sport' column
    df["sport"] = df.apply(calculate_sport_calories, axis=1)
    df.drop(columns=["Sport"], inplace=True)

    # Clean and format all column names
    new_columns = {}
    for col in df.columns:
        new_col = normalize_food_names(col)
        new_columns[col] = new_col
    df.rename(columns=new_columns, inplace=True)


    print("--- Feature Building Complete ---")
    return df


def main():
    """
    Orchestrates the data processing to build features.
    """
    print("--- Starting Data Processing ---")

    # Load data and calculate nutritional features
    features_df = data_processor.load_and_process_data(
        journal_path="data/processed_journal.csv",
        variables_path="data/variables.csv",
    )

    # Build features
    features_df = build_features(features_df)

    print("Successfully generated features DataFrame.")
    print(f"Shape: {features_df.shape}")
    print("\nFirst 5 rows of the features:")
    print(features_df.head())

    # Save the processed data
    features_df.to_csv("data/features.csv")
    print("\nSaved features to 'data/features.csv'")

    return features_df


if __name__ == "__main__":
    main()
