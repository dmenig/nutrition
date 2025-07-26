from typing import Dict


# MET values for various activities. This is a sample and will be expanded.
MET_VALUES: Dict[str, float] = {
    "running": 10.0,
    "cycling": 8.0,
    "walking": 3.5,
}


def WALKING_CALORIES(
    duration_minutes: float,
    weight_kg: float,
    distance_meters: float,
    additional_weight_kg: float,
) -> float:
    """
    Calculates the calories burned during walking.
    The parameters 'distance_meters' and 'additional_weight_kg' are placeholders.
    """
    met_value = MET_VALUES.get("walking", 3.5)
    return met_value * weight_kg * (duration_minutes / 60.0)


def RUNNING_CALORIES(
    duration_minutes: float,
    weight_kg: float,
    distance_meters: float,
    additional_weight_kg: float,
) -> float:
    """
    Calculates the calories burned from running.
    The parameters 'distance_meters' and 'additional_weight_kg' are placeholders.
    """
    met_value = MET_VALUES.get("running", 10.0)
    return met_value * weight_kg * (duration_minutes / 60.0)


def CYCLING_CALORIES(
    duration_minutes: float,
    weight_kg: float,
    distance_meters: float,
    additional_weight_kg: float,
) -> float:
    """
    Calculates the calories burned from cycling.
    The parameters 'distance_meters' and 'additional_weight_kg' are placeholders.
    """
    met_value = MET_VALUES.get("cycling", 8.0)
    return met_value * weight_kg * (duration_minutes / 60.0)


# Whitelist of functions that can be called from the sport formula.
# The keys are the function names in the formula (case-insensitive).
# The values are the actual function objects.
SPORT_FUNCTIONS = {
    "WALKING_CALORIES": WALKING_CALORIES,
    "RUNNING_CALORIES": RUNNING_CALORIES,
    "CYCLING_CALORIES": CYCLING_CALORIES,
}


def evaluate_sport_formula(formula: str, weight: float) -> float:
    """
    Evaluates a sport formula string.
    """
    if not formula or formula.isspace():
        return 0.0

    # Prepare the context for safe evaluation
    context = {
        "WEIGHT": weight,
        **SPORT_FUNCTIONS,
    }

    # Evaluate the formula using a safe evaluator
    from utils import SafeSportFormulaEvaluator
    import ast

    evaluator = SafeSportFormulaEvaluator(context)
    node = ast.parse(formula, mode="eval")
    return evaluator.visit(node.body)
