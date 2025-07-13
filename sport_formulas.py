from typing import Dict


# MET values for various activities. This is a sample and will be expanded.
MET_VALUES: Dict[str, float] = {
    "running": 10.0,
    "cycling": 8.0,
    "walking": 3.5,
}


def WALKING_CALORIES(
    duration_minutes: float, steps: float, incline_percent: float, weight_kg: float
) -> float:
    """
    Calculates the calories burned during walking.
    The parameters 'steps' and 'incline_percent' are placeholders.
    """
    met_value = MET_VALUES.get("walking", 3.5)
    return (met_value * 3.5 * weight_kg) / 200 * duration_minutes


def RUNNING_CALORIES(
    duration_minutes: float, distance_km: float, avg_speed_kmh: float, weight_kg: float
) -> float:
    """
    Calculates the calories burned from running.
    The parameters 'distance_km' and 'avg_speed_kmh' are placeholders.
    """
    met_value = MET_VALUES.get("running", 10.0)
    return (met_value * 3.5 * weight_kg) / 200 * duration_minutes


def CYCLING_CALORIES(
    duration_minutes: float, distance_km: float, avg_speed_kmh: float, weight_kg: float
) -> float:
    """
    Calculates the calories burned from cycling.
    The parameters 'distance_km' and 'avg_speed_kmh' are placeholders.
    """
    met_value = MET_VALUES.get("cycling", 8.0)
    return (met_value * 3.5 * weight_kg) / 200 * duration_minutes


# Whitelist of functions that can be called from the sport formula.
# The keys are the function names in the formula (case-insensitive).
# The values are the actual function objects.
SPORT_FUNCTIONS = {
    "walking_calories": WALKING_CALORIES,
    "running_calories": RUNNING_CALORIES,
    "cycling_calories": CYCLING_CALORIES,
}
