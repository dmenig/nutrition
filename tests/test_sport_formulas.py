import pytest
from sport_formulas import (
    WALKING_CALORIES,
    RUNNING_CALORIES,
    CYCLING_CALORIES,
)
from sport_formulas import evaluate_sport_formula

# Test cases for individual sport formula functions.
# Each tuple contains: (function, args, expected_result)
# args are (duration_minutes, weight_kg, param3, param4)
FORMULA_TEST_CASES = [
    # WALKING_CALORIES
    (WALKING_CALORIES, (60, 7000, 3, 70), 257.25),
    (WALKING_CALORIES, (0, 0, 0, 70), 0.0),
    # RUNNING_CALORIES
    (RUNNING_CALORIES, (30, 5, 10, 75), 393.75),
    (RUNNING_CALORIES, (0, 0, 0, 75), 0.0),
    # CYCLING_CALORIES
    (CYCLING_CALORIES, (45, 15, 20, 80), 504.0),
    (CYCLING_CALORIES, (0, 0, 0, 80), 0.0),
]


@pytest.mark.parametrize("func, args, expected", FORMULA_TEST_CASES)
def test_sport_formula_functions(func, args, expected):
    """
    Tests the individual sport formula functions with various inputs.
    The extra parameters (steps, distance, etc.) are placeholders and do not
    affect the current calculation, but are included for signature correctness.
    """
    result = func(*args)
    assert result == pytest.approx(expected)


def test_evaluate_sport_formula_with_commas_and_functions():
    """
    Tests evaluate_sport_formula with a complex formula including commas
    as decimal separators and multiple function calls, verifying the fix.
    """
    formula = "15*8 + 2*WALKING_CALORIES(duration_minutes=10, weight_kg=70, steps=700, incline_percent=4) + WALKING_CALORIES(duration_minutes=27, weight_kg=70, steps=2800, incline_percent=2)"
    weight = 70.0  # Dummy weight value
    expected_calories = 321.5125  # Calculated: 120 + 2*42.875 + 115.7625

    result = evaluate_sport_formula(formula, weight)
    assert result == pytest.approx(expected_calories)
