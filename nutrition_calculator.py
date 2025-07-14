import csv
import ast
import pandas as pd
from utils import SafeFormulaEvaluator, strip_accents


def get_nutrient_context(nutrient: str, variables_df: pd.DataFrame) -> dict:
    """
    Extracts nutrient values for each food from the DataFrame.

    Args:
        nutrient: The specific nutrient to extract (e.g., 'Kcal').
        variables_df: Pandas DataFrame containing nutrient variables.

    Returns:
        A dictionary mapping food names to their corresponding nutrient values.
    """
    nutrient_context = {}
    for _, row in variables_df.iterrows():
        food_name = strip_accents(str(row["Nom"]).lower().strip()).replace("_", " ")
        if nutrient in row and pd.notna(row[nutrient]):
            nutrient_context[food_name] = float(row[nutrient])
    return nutrient_context


def calculate_nutrient_from_formula_with_context(
    formula: str, nutrient_context: dict, nutrient: str
) -> float:
    """
    Calculates the total nutrient value from a food formula string using a pre-built nutrient context.

    Args:
        formula: A string representing the food formula (e.g., "100 * pain + 50 * fromage").
        nutrient_context: A dictionary mapping food names to their corresponding nutrient values.
        nutrient: The specific nutrient to calculate (e.g., 'Kcal').

    Returns:
        The calculated total nutrient value.
    """
    # Normalize food_formula before parsing
    # Replace commas with dots for decimal compatibility
    normalized_food_formula = strip_accents(formula.lower()).replace(",", ".")
    # Parse the expression into an AST
    node = ast.parse(normalized_food_formula, mode="eval")
    # Evaluate the AST using the safe evaluator with the prepared context
    evaluator = SafeFormulaEvaluator(context=nutrient_context)
    try:
        result = evaluator.visit(node.body)
    except NameError as e:
        # Extract the variable name from the NameError message
        variable_name = str(e).split("'")[1]
        raise ValueError(
            f"Name '{variable_name}' is not defined in the given context."
        ) from e
    return result


def calculate_nutrient_from_formula(
    food_formula: str, nutrient: str, variables_file_path: str = "data/variables.csv"
) -> float:
    """
    Calculates the total nutrient value from a food formula string.

    Args:
        food_formula: A string representing the food formula (e.g., "100 * pain + 50 * fromage").
        nutrient: The specific nutrient to calculate (e.g., 'Kcal').
        variables_file_path: Path to the CSV file containing nutrient variables.

    Returns:
        The calculated total nutrient value.
    """
    variables_df = pd.read_csv(variables_file_path)
    nutrient_context = get_nutrient_context(nutrient, variables_df)
    return calculate_nutrient_from_formula_with_context(
        food_formula, nutrient_context, nutrient
    )
