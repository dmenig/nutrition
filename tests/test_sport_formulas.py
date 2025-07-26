import pytest
from sport_formulas import (
    WALKING_CALORIES,
    RUNNING_CALORIES,
    CYCLING_CALORIES,
)
from sport_formulas import evaluate_sport_formula

# Test cases for individual sport formula functions.
# Each tuple contains: (function, args, expected_result)
# args are (duration_minutes, weight_kg, distance_meters, additional_weight_kg)
FORMULA_TEST_CASES = [
    # WALKING_CALORIES
    (WALKING_CALORIES, (60, 70, 3000, 5), 245.0),
    (WALKING_CALORIES, (0, 70, 0, 0), 0.0),
    # RUNNING_CALORIES
    (RUNNING_CALORIES, (30, 75, 5000, 0), 375.0),
    (RUNNING_CALORIES, (0, 75, 0, 0), 0.0),
    # CYCLING_CALORIES
    (CYCLING_CALORIES, (45, 80, 15000, 0), 480.0),
    (CYCLING_CALORIES, (0, 80, 0, 0), 0.0),
]


@pytest.mark.parametrize("func, args, expected", FORMULA_TEST_CASES)
def test_sport_formula_functions(func, args, expected):
    """
    Tests the individual sport formula functions with various inputs.
    The extra parameters (distance_meters, additional_weight_kg) are placeholders and do not
    affect the current calculation, but are included for signature correctness.
    """
    result = func(*args)
    assert result == pytest.approx(expected)


def test_evaluate_sport_formula_with_commas_and_functions():
    """
    Tests evaluate_sport_formula with a complex formula including commas
    as decimal separators and multiple function calls, verifying the fix.
    """
    formula = "15*8 + 2*WALKING_CALORIES(10, 70, 700, 4) + WALKING_CALORIES(27, 70, 2800, 2)"
    weight = 70.0  # Dummy weight value
    expected_calories = 311.9167 # Calculated: 120 + 2*40.8333 + 110.25

    result = evaluate_sport_formula(formula, weight)
    assert result == pytest.approx(expected_calories, 0.01)

def test_evaluate_sport_formula_with_positional_args():
    """
    Tests that the evaluate_sport_formula function correctly processes
    a formula with positional arguments, including the WEIGHT placeholder.
    """
    formula = "WALKING_CALORIES(7, WEIGHT, 700, 3)"
    weight = 70.0
    # Expected: 3.5 (MET) * 70 (weight) * (7 / 60) (duration) = 28.5833
    expected_calories = 28.5833
    result = evaluate_sport_formula(formula, weight)
    assert result == pytest.approx(expected_calories, 0.0001)
