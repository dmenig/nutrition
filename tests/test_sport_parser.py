import pytest
from sport_formulas import evaluate_sport_formula

# Test cases for the sport formula evaluator.
# Each tuple contains: (formula_string, weight_kg, expected_result)
TEST_CASES = [
    # Simple cases
    ("WALKING_CALORIES(60, 7000, 3, 70)", 70, 257.25),
    ("RUNNING_CALORIES(30, 5, 10, 75)", 75, 393.75),
    ("CYCLING_CALORIES(45, 15, 20, 80)", 80, 504.0),
    # Case with arithmetic
    (
        "WALKING_CALORIES(30, 3000, 1, 68) + RUNNING_CALORIES(15, 3, 12, 68)",
        68,
        303.45,
    ),
    # Formula with different weight
    ("WALKING_CALORIES(60, 8000, 2, 75)", 75, 275.625),
    # Formula with zero values
    ("WALKING_CALORIES(0, 70, 0, 0)", 70, 0.0),
]

# Test cases that are expected to raise an error.
# Each tuple contains: (formula_string, weight_kg, error_type, error_message_substring)
ERROR_CASES = [
    # Undefined function
    ("UNDEFINED_FUNCTION(60, 70)", 70, NameError, "not allowed"),
    # Undefined variable
    (
        "WALKING_CALORIES(60, UNDEFINED_VAR, 7000, 3)",
        70,
        NameError,
        "Variable 'UNDEFINED_VAR' is not defined",
    ),
    # Incorrect number of arguments
    ("WALKING_CALORIES(60, 70)", 70, TypeError, "required positional arguments"),
    # Malformed expression
    ("WALKING_CALORIES(60, 70, 7000, 3) +", 70, SyntaxError, "invalid syntax"),
    # Unsafe code
    (
        "__import__('os').system('echo unsafe')",
        70,
        NameError,
        "Indirect function calls are not allowed.",
    ),
]


@pytest.mark.parametrize("formula, weight, expected", TEST_CASES)
def test_evaluate_sport_formula_valid(formula, weight, expected):
    """Tests valid sport formulas."""
    result = evaluate_sport_formula(formula, weight)
    assert result == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("formula, weight, error, msg", ERROR_CASES)
def test_evaluate_sport_formula_invalid(formula, weight, error, msg):
    """Tests invalid or unsafe sport formulas."""
    with pytest.raises(error, match=msg):
        evaluate_sport_formula(formula, weight)


def test_evaluate_sport_formula_empty_and_none():
    """Tests that empty, whitespace, or None formulas return 0.0."""
    assert evaluate_sport_formula("", 70) == 0.0
    assert evaluate_sport_formula("   ", 70) == 0.0
    assert evaluate_sport_formula(None, 70) == 0.0
