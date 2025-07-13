import pandas as pd
import re
import unicodedata
import ast
import operator as op
import sport_formulas as sf
import inspect


def get_clean_food_name(name: str) -> str:
    """Cleans a food name by removing accents, lowercasing, and removing non-alphanumeric characters."""
    name = name.replace("_", " ")
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
    name = re.sub(r"[^a-z0-9\s]", "", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def evaluate_expression(expression_str: str, context: dict) -> float:
    """
    Evaluates a string expression by substituting variables from a context and using eval().
    """
    if not isinstance(expression_str, str):
        return 0.0

    expression = expression_str.replace(",", ".").replace("_", " ")

    # Substitute all known variables/food names from the context
    # Sort by length to avoid partial replacements (e.g., 'pain' before 'pain complet')
    for name in sorted(context.keys(), key=len, reverse=True):
        expression = re.sub(
            r"\b" + re.escape(name) + r"\b",
            str(context[name]),
            expression,
            flags=re.IGNORECASE,
        )

    # Replace any remaining words with 0 to handle unlisted items gracefully
    expression = re.sub(r"[a-zA-Z_]+", "0", expression)

    return eval(expression, {"__builtins__": {}}, {})


def evaluate_sport_expression(expression_str: str, context: dict) -> float:
    """
    Evaluates a sport-related expression using a safe context.
    """
    if not isinstance(expression_str, str):
        return 0.0

    # Prepare the expression for evaluation
    expression = expression_str.strip().lstrip("=").replace(",", ".")

    # The context should already contain 'WEIGHT' and sport functions
    # No need to substitute other text, as only defined functions and WEIGHT should be present
    try:
        return eval(expression, {"__builtins__": {}}, context)
    except (SyntaxError, NameError, TypeError, ValueError) as e:
        print(f"Could not evaluate sport expression '{expression_str}': {e}")
        return 0.0


def main():
    """
    Main function to run the nutrition calculation process.
    """
    # 1. Read Nutritional Data
    nutritional_data = pd.read_csv("data/variables.csv")

    nutritional_data["key"] = nutritional_data["Nom"].apply(get_clean_food_name)
    nutritional_data.drop_duplicates(subset="key", keep="first", inplace=True)
    nutrition_dict = nutritional_data.set_index("key").to_dict("index")

    # 2. Process Journal Data
    journal_df = pd.read_csv("data/processed_journal.csv")
    results = []

    # 3. Parse and Calculate
    for index, row in journal_df.iterrows():
        nourriture_str = row["Nourriture"]
        result_row = row.to_dict()

        # --- Nutrient Calculation ---
        nutrient_cols = [
            col for col in nutritional_data.columns if col not in ["Nom", "key"]
        ]
        for nutrient in nutrient_cols:
            # Create a specific context for the current nutrient
            nutrient_context = {
                get_clean_food_name(food): data.get(nutrient, 0)
                for food, data in nutrition_dict.items()
            }
            total_nutrient = evaluate_expression(nourriture_str, nutrient_context)
            result_row[f"Total {nutrient.replace(' / 100g', '')}"] = total_nutrient

        # --- Sport and Weight Calculation ---
        sport_str = row.get("Sport")
        weight_str = row.get("Pds")
        total_sport_calories = 0

        # Create a context for sport and weight formulas
        sport_context = {
            func_name.upper(): func_obj
            for func_name, func_obj in inspect.getmembers(sf, inspect.isfunction)
        }
        if pd.notna(weight_str):
            # Evaluate weight first, as it's needed for sport calories
            weight_kg = float(weight_str)  # Pds should now be a clean number
            sport_context["WEIGHT"] = weight_kg
            result_row["Pds"] = weight_kg

            if pd.notna(sport_str):
                total_sport_calories = evaluate_sport_expression(
                    sport_str, sport_context
                )

        # Add sport and net calories to results
        result_row["Total Sport Calories"] = total_sport_calories
        result_row["Net Kcal"] = result_row.get("Total Kcal", 0) - total_sport_calories

        results.append(result_row)

    # 4. Output Results
    results_df = pd.DataFrame(results)
    original_cols = list(journal_df.columns)
    calculated_cols = [col for col in results_df.columns if col not in original_cols]
    results_df = results_df[original_cols + calculated_cols]

    # 5. Save Results
    results_df.to_csv("data/calculated_nutrition.csv", index=False)

    print(
        "Nutritional calculation complete. Results saved to 'data/calculated_nutrition.csv'"
    )


if __name__ == "__main__":
    main()
