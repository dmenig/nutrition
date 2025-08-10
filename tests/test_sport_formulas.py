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
    # WALKING_CALORIES: speed = 3km/h, MET = 2.438, weight = 75 -> 2.438 * 75 * 1 = 182.8846
    (WALKING_CALORIES, (60, 70, 3000, 5), 182.884615),
    # RUNNING_CALORIES: speed = 10km/h, MET = 10.27, weight = 75 -> 10.27 * 75 * 0.5 = 385.2273
    (RUNNING_CALORIES, (30, 75, 5000, 0), 385.227273),
    # CYCLING_CALORIES: speed = 20km/h, MET = 6.5, weight = 80 -> 6.5 * 80 * 0.75 = 390.0
    (CYCLING_CALORIES, (45, 80, 15000, 0), 390.0),
    # Test cases for duration <= 0
    (WALKING_CALORIES, (0, 70, 0, 0), ValueError),
    (RUNNING_CALORIES, (0, 75, 0, 0), ValueError),
    (CYCLING_CALORIES, (0, 80, 0, 0), ValueError),
]


@pytest.mark.parametrize("func, args, expected", FORMULA_TEST_CASES)
def test_sport_formula_functions(func, args, expected):
    """
    Tests the individual sport formula functions with various inputs.
    The extra parameters (distance_meters, additional_weight_kg) are placeholders and do not
    affect the current calculation, but are included for signature correctness.
    """
    if expected == ValueError:
        with pytest.raises(ValueError):
            func(*args)
    else:
        result = func(*args)
        assert result == pytest.approx(expected)


def test_evaluate_sport_formula_with_commas_and_functions():
    """
    Tests evaluate_sport_formula with a complex formula including commas
    as decimal separators and multiple function calls, verifying the fix.
    """
    formula = (
        "15*8 + 2*WALKING_CALORIES(10, 70, 700, 4) + WALKING_CALORIES(27, 70, 2800, 2)"
    )
    expected_calories = 347.04

    result = evaluate_sport_formula(formula)
    assert result == pytest.approx(expected_calories, rel=1e-2)


def test_evaluate_sport_formula_with_positional_args():
    """
    Tests that the evaluate_sport_formula function correctly processes
    a formula with positional arguments, including the WEIGHT placeholder.
    """
    formula = "WALKING_CALORIES(7, 70, 700, 3)"
    # Expected: speed = 6km/h, MET = 4.8, weight = 73 -> 4.8 * 73 * (7/60) = 40.88
    expected_calories = 37.284
    result = evaluate_sport_formula(formula)
    assert result == pytest.approx(expected_calories, rel=1e-2)
