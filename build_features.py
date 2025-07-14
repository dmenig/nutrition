import data_processor
import sport_formulas
import nutrition_calculator
import pandas as pd
import ast


def calculate_sport_calories(row: pd.Series) -> float:
    """
    Calculates calories burned from a sport formula string.

    Args:
        row: A pandas Series representing a row of the DataFrame.
             It must contain 'Sport' and 'Pds' columns.

    Returns:
        The calculated calories burned, or 0 if the formula is invalid.
    """
    sport_formula = row["Sport"]
    weight = row["Pds"]

    if not isinstance(sport_formula, str) or not sport_formula.strip():
        return 0.0

    try:
        node = ast.parse(sport_formula, mode="eval")

        if not isinstance(node.body, ast.Call):
            raise ValueError("Sport formula must be a single function call.")

        func_name = node.body.func.id.lower()
        if func_name not in sport_formulas.SPORT_FUNCTIONS:
            raise ValueError(f"Unknown sport function: {func_name}")

        # Extract keyword arguments from the formula string
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.body.keywords}

        # Add weight to the arguments
        kwargs["weight_kg"] = weight

        sport_function = sport_formulas.SPORT_FUNCTIONS[func_name]

        return sport_function(**kwargs)

    except (ValueError, SyntaxError, TypeError, KeyError) as e:
        print(
            f"Warning: Could not parse sport formula '{sport_formula}': {e}. Setting to 0."
        )
        return 0.0


def main():
    """
    Orchestrates the data processing to build features.
    """
    print("--- Starting Data Processing ---")

    # Load data and calculate nutritional features
    features_df = data_processor.load_and_process_data()

    print("\n--- Calculating Sport Calories ---")
    if "Sport" in features_df.columns and "Pds" in features_df.columns:
        features_df["calories_sport"] = features_df.apply(
            calculate_sport_calories, axis=1
        )
        print("Successfully calculated sport calories.")
        features_df.drop(columns=["Sport"], inplace=True)
    else:
        print("'Sport' or 'Pds' column not found, skipping sport calorie calculation.")
        features_df["calories_sport"] = 0.0

    # Ensure column names are consistently formatted
    features_df.columns = [
        str(col).lower().replace(" ", "_").replace("/", "_")
        for col in features_df.columns
    ]

    print("\n--- Feature Building Complete ---")
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
