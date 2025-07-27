from typing import Dict, List, Tuple
import ast
import numpy as np
from utils import SafeSportFormulaEvaluator


# MET values for various activities, dependent on speed in km/h.
# The format is a list of (speed_kmh, met_value) tuples, sorted by speed.
# Data is sourced from the Compendium of Physical Activities and scientific publications.
MET_VALUES: Dict[str, List[Tuple[float, float]]] = {
    "walking": [
        (2.7, 2.3),
        (4.0, 2.9),
        (4.8, 3.3),
        (5.5, 3.6),
        (6.4, 5.0),
        (8.0, 6.5),
    ],
    "running": [
        (8.0, 8.0),
        (8.4, 9.0),
        (9.7, 10.0),
        (10.8, 11.0),
        (11.3, 11.5),
        (12.1, 12.5),
        (12.9, 13.5),
        (13.8, 14.0),
        (14.5, 15.0),
        (16.1, 16.0),
        (17.5, 18.0),
    ],
    "cycling": [
        (16.0, 4.0),
        (19.2, 6.0),
        (22.4, 8.0),
        (25.6, 10.0),
        (30.6, 12.0),
        (100.0, 16.0),
    ],
}


def get_met_value(activity: str, speed_kmh: float) -> float:
    """
    Gets the MET value for an activity based on speed, using linear interpolation.
    If speed is outside the defined range, it clamps to the min/max value.
    """

    met_table = MET_VALUES[activity]
    speeds, mets = zip(*met_table)

    if speed_kmh <= speeds[0]:
        return mets[0]
    if speed_kmh >= speeds[-1]:
        return mets[-1]

    return float(np.interp(speed_kmh, speeds, mets))


def WALKING_CALORIES(
    duration_minutes: float,
    weight_kg: float,
    distance_meters: float,
    additional_weight_kg: float,
) -> float:
    """
    Calculates the calories burned during walking based on speed.
    """
    if duration_minutes <= 0:
        raise ValueError("Duration must be greater than 0")

    speed_kmh = (distance_meters / 1000.0) / (duration_minutes / 60.0)
    met_value = get_met_value("walking", speed_kmh)
    total_weight = weight_kg + additional_weight_kg
    return met_value * total_weight * (duration_minutes / 60.0)


def RUNNING_CALORIES(
    duration_minutes: float,
    weight_kg: float,
    distance_meters: float,
    additional_weight_kg: float,
) -> float:
    """
    Calculates the calories burned from running based on speed.
    """
    if duration_minutes <= 0:
        raise ValueError("Duration must be greater than 0")

    speed_kmh = (distance_meters / 1000.0) / (duration_minutes / 60.0)
    met_value = get_met_value("running", speed_kmh)
    total_weight = weight_kg + additional_weight_kg
    return met_value * total_weight * (duration_minutes / 60.0)


def CYCLING_CALORIES(
    duration_minutes: float,
    weight_kg: float,
    distance_meters: float,
    additional_weight_kg: float,
) -> float:
    """
    Calculates the calories burned from cycling based on speed.
    """
    if duration_minutes <= 0:
        raise ValueError("Duration must be greater than 0")

    speed_kmh = (distance_meters / 1000.0) / (duration_minutes / 60.0)
    met_value = get_met_value("cycling", speed_kmh)
    total_weight = weight_kg + additional_weight_kg
    return met_value * total_weight * (duration_minutes / 60.0)


# Whitelist of functions that can be called from the sport formula.
# The keys are the function names in the formula (case-insensitive).
# The values are the actual function objects.
SPORT_FUNCTIONS = {
    "WALKING_CALORIES": WALKING_CALORIES,
    "RUNNING_CALORIES": RUNNING_CALORIES,
    "CYCLING_CALORIES": CYCLING_CALORIES,
}


def evaluate_sport_formula(formula: str) -> float:
    """
    Evaluates a sport formula string.
    """
    if not formula or formula.isspace():
        return 0.0

    # Prepare the context for safe evaluation
    context = {
        **SPORT_FUNCTIONS,
    }

    # Evaluate the formula using a safe evaluator
    evaluator = SafeSportFormulaEvaluator(context)
    node = ast.parse(formula, mode="eval")
    return evaluator.visit(node.body)
