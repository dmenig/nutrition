import pytest
import os
import csv
from nutrition_calculator import calculate_nutrient_from_formula
import pandas as pd
from utils import normalize_food_names

# Define the path for the temporary variables.csv file
TEMP_VARIABLES_CSV = "tests/temp_variables.csv"


@pytest.fixture(scope="module")
def setup_teardown_variables_csv():
    """
    Sets up a temporary variables.csv file for testing and tears it down afterwards.
    """
    # Create a dummy variables.csv file for testing
    with open(TEMP_VARIABLES_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Nom", "Kcal", "Protéine", "Glucide", "Lipide"])
        writer.writerow(["Pomme", "52", "0.3", "14", "0.2"])
        writer.writerow(["Pomme verte", "60", "0.4", "15", "0.1"])
        writer.writerow(["Banane", "89", "1.1", "23", "0.3"])
        writer.writerow(["Banane_plantain", "89", "1.1", "23", "0.3"])
        writer.writerow(["Lait", "42", "3.4", "4.8", "1"])
        writer.writerow(["Pain", "265", "9", "49", "3.2"])
        writer.writerow(["Fromage", "402", "25", "1.3", "33"])

    # Yield control to the tests
    yield

    # Teardown: Remove the dummy variables.csv file
    os.remove(TEMP_VARIABLES_CSV)


def test_simple_formula_one_ingredient(setup_teardown_variables_csv):
    """
    Tests a simple formula with one ingredient for Kcal and Protéine.
    Formula: "Pomme * 100"
    Expected: Pomme (52 Kcal, 0.3 Protéine) * 100 = 5200 Kcal, 30 Protéine
    """
    formula = "Pomme * 100"

    # Test Kcal
    result_kcal = calculate_nutrient_from_formula(formula, "Kcal", TEMP_VARIABLES_CSV)
    assert result_kcal == pytest.approx(5200.0)

    # Test Protéine
    result_protein = calculate_nutrient_from_formula(
        formula, "Protéine", TEMP_VARIABLES_CSV
    )
    assert result_protein == pytest.approx(30.0)


def test_complex_formula_multiple_ingredients(setup_teardown_variables_csv):
    """
    Tests a complex formula with multiple ingredients and operations.
    Formula: "(Pomme * 50) + (Banane * 200) - (Lait * 10)"
    Expected Kcal: (52 * 50) + (89 * 200) - (42 * 10) = 2600 + 17800 - 420 = 19980
    Expected Protéine: (0.3 * 50) + (1.1 * 200) - (3.4 * 10) = 15 + 220 - 34 = 201
    """
    formula = "(Pomme * 50) + (Banane * 200) - (Lait * 10)"

    # Test Kcal
    result_kcal = calculate_nutrient_from_formula(formula, "Kcal", TEMP_VARIABLES_CSV)
    assert result_kcal == pytest.approx(19980.0)

    # Test Protéine
    result_protein = calculate_nutrient_from_formula(
        formula, "Protéine", TEMP_VARIABLES_CSV
    )
    assert result_protein == pytest.approx(201.0)


def test_food_item_not_in_variables_file(setup_teardown_variables_csv):
    """
    Tests a scenario where a food item in the formula is not present in the variables file.
    This should raise a ValueError.
    """
    formula = "Orange * 100"
    with pytest.raises(
        ValueError,
        match=r"Name 'Orange' is not defined in the given context.",
    ):
        calculate_nutrient_from_formula(formula, "Kcal", TEMP_VARIABLES_CSV)


def test_different_nutrients(setup_teardown_variables_csv):
    """
    Tests calculation for different nutrients (Glucide, Lipide).
    Formula: "Pain * 10 + Fromage * 5"
    Expected Glucide: (49 * 10) + (1.3 * 5) = 490 + 6.5 = 496.5
    Expected Lipide: (3.2 * 10) + (33 * 5) = 32 + 165 = 197
    """
    formula = "Pain * 10 + Fromage * 5"

    # Test Glucide
    result_glucide = calculate_nutrient_from_formula(
        formula, "Glucide", TEMP_VARIABLES_CSV
    )
    assert result_glucide == pytest.approx(496.5)

    # Test Lipide
    result_lipide = calculate_nutrient_from_formula(
        formula, "Lipide", TEMP_VARIABLES_CSV
    )
    assert result_lipide == pytest.approx(197.0)


def test_formula_with_division(setup_teardown_variables_csv):
    """
    Tests a formula involving division.
    Formula: "Pomme * 200 / 2"
    Expected Kcal: (52 * 200) / 2 = 10400 / 2 = 5200
    """
    formula = "Pomme * 200 / 2"
    result_kcal = calculate_nutrient_from_formula(formula, "Kcal", TEMP_VARIABLES_CSV)
    assert result_kcal == pytest.approx(5200.0)


def test_formula_with_addition_and_subtraction(setup_teardown_variables_csv):
    """
    Tests a formula with only addition and subtraction.
    Formula: "Pomme * 100 + Banane * 50 - Lait * 10"
    Expected Kcal: (52 * 100) + (89 * 50) - (42 * 10) = 5200 + 4450 - 420 = 9230
    """
    formula = "Pomme * 100 + Banane * 50 - Lait * 10"
    result_kcal = calculate_nutrient_from_formula(formula, "Kcal", TEMP_VARIABLES_CSV)
    assert result_kcal == pytest.approx(9230.0)


def test_formula_with_parentheses_and_mixed_operations(setup_teardown_variables_csv):
    """
    Tests a formula with parentheses and mixed operations to ensure correct order of operations.
    Formula: "Pain * (10 + 5) - Fromage * 2"
    Expected Kcal: 265 * (15) - 402 * 2 = 3975 - 804 = 3171
    """
    formula = "Pain * (10 + 5) - Fromage * 2"
    result_kcal = calculate_nutrient_from_formula(formula, "Kcal", TEMP_VARIABLES_CSV)
    assert result_kcal == pytest.approx(3171.0)


def test_formula_with_underscores_in_food_name(setup_teardown_variables_csv):
    """
    Tests that a formula with underscores in a food name is correctly interpreted.
    Formula: "Pomme_verte * 100"
    Expected Kcal: 60 * 100 = 6000
    """
    formula = "Pomme_verte * 100"
    result_kcal = calculate_nutrient_from_formula(formula, "Kcal", TEMP_VARIABLES_CSV)
    assert result_kcal == pytest.approx(6000.0)


def test_interchangeable_spaces_and_underscores(setup_teardown_variables_csv):
    """
    Tests that food names with spaces and underscores are handled interchangeably.

    Formula: "Pomme_verte * 100"
    Expected Kcal: 60 * 100 = 6000 (from "Pomme verte" in CSV)
    """
    formula = "Pomme_verte * 100"
    result_kcal = calculate_nutrient_from_formula(formula, "Kcal", TEMP_VARIABLES_CSV)
    assert result_kcal == pytest.approx(6000.0)


def test_normalization_and_calculation():
    # 1. Create a dummy variables.csv for testing
    variables_data = {
        "Nom": ["Caviar d'aubergine", "Autre chose"],
        "Calories / 100g": [150, 200],
        "Proteines / 100g": [5, 10],
    }
    variables_df = pd.DataFrame(variables_data)
    
    # Normalize the 'Nom' column
    variables_df["Nom"] = variables_df["Nom"].apply(normalize_food_names)
    
    os.makedirs("data", exist_ok=True)
    variables_csv_path = "data/variables.csv"
    variables_df.to_csv(variables_csv_path, index=False)

    # 2. Test the calculation with a formula using the un-normalized name
    formula = "100 * caviar_d_aubergine"

    # We need to use the normalized file for calculation
    calculated_kcal = calculate_nutrient_from_formula(
        formula, "Calories / 100g", variables_file_path=variables_csv_path
    )

    assert calculated_kcal == 15000.0

    # 3. Clean up the dummy files
    os.remove(variables_csv_path)
