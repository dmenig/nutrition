import pytest
from sport_formulas import evaluate_sport_formula

# Test cases for the sport formula evaluator.
# Each tuple contains: (formula_string, weight_kg, expected_result)
TEST_CASES = [
    # Simple cases
    ("WALKING_CALORIES(60, 70, 7000, 3)", 406.0625),
    ("RUNNING_CALORIES(30, 75, 5000, 0)", 385.2272),
    ("CYCLING_CALORIES(45, 80, 15000, 0)", 390.0),
    # Case with arithmetic
    (
        "WALKING_CALORIES(30, 68, 3000, 1) + RUNNING_CALORIES(15, 68, 3000, 12)",
        398.5333,
    ),
    # Formula with different weight
    ("WALKING_CALORIES(60, 75, 8000, 2)", 500.5),
    # Formula with zero values
    ("WALKING_CALORIES(0, 70, 0, 0)", 0.0),
]

# Test cases that are expected to raise an error.
# Each tuple contains: (formula_string, weight_kg, error_type, error_message_substring)
ERROR_CASES = [
    # Undefined function
    ("UNDEFINED_FUNCTION(60, 70)", NameError, "not allowed"),
    # Undefined variable
    (
        "WALKING_CALORIES(60, UNDEFINED_VAR, 7000, 3)",
        ValueError,
        "Name 'UNDEFINED_VAR' is not defined",
    ),
    # Incorrect number of arguments
    ("WALKING_CALORIES(60, 70)", TypeError, "missing 2 required positional arguments"),
    # Malformed expression
    ("WALKING_CALORIES(60, 70, 7000, 3) +", SyntaxError, "invalid syntax"),
    # Unsafe code
    (
        "__import__('os').system('echo unsafe')",
        NameError,
        "Indirect function calls are not allowed.",
    ),
]


@pytest.mark.parametrize("formula, expected", TEST_CASES)
def test_evaluate_sport_formula_valid(formula, expected):
    """Tests valid sport formulas."""
    result = evaluate_sport_formula(formula)
    assert result == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("formula, error, msg", ERROR_CASES)
def test_evaluate_sport_formula_invalid(formula, error, msg):
    """Tests invalid or unsafe sport formulas."""
    with pytest.raises(error, match=msg):
        evaluate_sport_formula(formula)


def test_evaluate_sport_formula_empty_and_none():
    """Tests that empty, whitespace, or None formulas return 0.0."""
    assert evaluate_sport_formula("") == 0.0
    assert evaluate_sport_formula("   ") == 0.0
    assert evaluate_sport_formula(None) == 0.0
