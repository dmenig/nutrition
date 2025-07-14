import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from build_features import main as build_features_main
from sport_formulas import WALKING_CALORIES, RUNNING_CALORIES


def test_build_features_end_to_end():
    """
    Comprehensive end-to-end test for the build_features script.
    This test mocks the file system and verifies the entire data processing pipeline
    from dummy input files to the final features DataFrame.
    """
    # 1. Create dummy dataframes in memory
    variables_df = pd.DataFrame(
        {
            "Nom": [
                "farine",
                "sucre",
                "pain complet",
                "fromage de chèvre",
                "poulet",
                "seven_up_mojito",
            ],
            "Calories / 100g": [364, 387, 259, 364, 239, 40],
            "Protéine": [10.3, 0, 13.4, 21.6, 27.0, 0],
            "Fat": [1.0, 0, 3.2, 29.8, 14.0, 0],
            "SFat": [0.2, 0, 0.7, 20.2, 3.8, 0],
            "Carbs": [76.3, 100, 41.3, 1.3, 0, 10.6],
            "Sugar": [0.3, 100, 5.4, 0.5, 0, 10.6],
            "Free sugar": [0, 100, 1.0, 0, 0, 10.6],
            "Fibres": [2.7, 0, 6.0, 0, 0, 0],
            "Sel": [0.01, 0, 1.4, 1.8, 0.2, 0.01],
            "Alcool": [0, 0, 0, 0, 0, 0.5],
            "Water": [12.0, 0, 35.0, 45.0, 60.0, 88.0],
        }
    )

    journal_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "Pds": [80.0, 80.2, 80.1, 79.9],
            "Nourriture": [
                "150 * (pain complet * 0.5 + fromage de chèvre * 0.5) + 200 * poulet",
                "100 * farine + 50 * sucre",
                "10 * seven_up_mojito + 0.682* (pain complet*0.8 + poulet*0.2)",
                "",  # Test empty food formula
            ],
            "Sport": [
                "15*8+ 2*WALKING_CALORIES(duration_minutes=750, steps=19, incline_percent=19, weight_kg=WEIGHT) + RUNNING_CALORIES(duration_minutes=1200, distance_km=10, avg_speed_kmh=25, weight_kg=WEIGHT)",
                "",
                "WALKING_CALORIES(duration_minutes=3600, steps=5, incline_percent=0, weight_kg=WEIGHT)",
                "10*10",  # Test simple sport formula
            ],
        }
    )

    # 2. Use `patch` to mock `pd.read_csv` and `df.to_csv`
    with (
        patch("pandas.read_csv") as mock_read_csv,
        patch("pandas.DataFrame.to_csv") as mock_to_csv,
    ):
        # Configure the mock to return different dataframes based on the file path
        def read_csv_side_effect(filepath):
            if "journal" in filepath:
                return journal_df.copy()
            elif "variables" in filepath:
                return variables_df.copy()
            return pd.DataFrame()

        mock_read_csv.side_effect = read_csv_side_effect

        # 3. Run the main script
        result_df = build_features_main()

    # 4. Assertions
    assert result_df is not None
    assert not result_df.empty

    # Assert column names are correctly formatted
    expected_columns = [
        "pds",
        "calories",
        "proteine",
        "fat",
        "sfat",
        "carbs",
        "sugar",
        "free_sugar",
        "fibres",
        "sel",
        "alcool",
        "water",
        "sport",
    ]
    # The actual column names are modified in the build_features script, so we need to check against the modified names
    # The actual column names are modified in the build_features script, so we need to check against the modified names
    assert set(result_df.columns) == set(expected_columns)

    # Pre-calculate expected values
    # Note: The division by 100 is crucial here as it was in the original implementation
    # of data_processor.py. We are testing the end-to-end result.

    # Row 1
    weight1 = 80.0
    pain_complet_ratio = 0.5
    fromage_chevre_ratio = 0.5
    calories1 = (
        150 * (259 * pain_complet_ratio + 364 * fromage_chevre_ratio) + 200 * 239
    ) / 100
    proteine1 = (
        150 * (13.4 * pain_complet_ratio + 21.6 * fromage_chevre_ratio) + 200 * 27.0
    ) / 100
    fat1 = (
        150 * (3.2 * pain_complet_ratio + 29.8 * fromage_chevre_ratio) + 200 * 14.0
    ) / 100
    sfat1 = (
        150 * (0.7 * pain_complet_ratio + 20.2 * fromage_chevre_ratio) + 200 * 3.8
    ) / 100
    carbs1 = (
        150 * (41.3 * pain_complet_ratio + 1.3 * fromage_chevre_ratio) + 200 * 0
    ) / 100
    sugar1 = (
        150 * (5.4 * pain_complet_ratio + 0.5 * fromage_chevre_ratio) + 200 * 0
    ) / 100
    free_sugar1 = (
        150 * (1.0 * pain_complet_ratio + 0 * fromage_chevre_ratio) + 200 * 0
    ) / 100
    fibres1 = (
        150 * (6.0 * pain_complet_ratio + 0 * fromage_chevre_ratio) + 200 * 0
    ) / 100
    sel1 = (
        150 * (1.4 * pain_complet_ratio + 1.8 * fromage_chevre_ratio) + 200 * 0.2
    ) / 100
    alcool1 = (
        150 * (0 * pain_complet_ratio + 0 * fromage_chevre_ratio) + 200 * 0
    ) / 100
    water1 = (
        150 * (35.0 * pain_complet_ratio + 45.0 * fromage_chevre_ratio) + 200 * 60.0
    ) / 100
    sport1 = (
        15 * 8
        + 2
        * WALKING_CALORIES(
            duration_minutes=750, steps=19, incline_percent=19, weight_kg=weight1
        )
        + RUNNING_CALORIES(
            duration_minutes=1200, distance_km=10, avg_speed_kmh=25, weight_kg=weight1
        )
    )

    # Row 2
    calories2 = (100 * 364 + 50 * 387) / 100
    proteine2 = (100 * 10.3 + 50 * 0) / 100
    fat2 = (100 * 1.0 + 50 * 0) / 100
    sfat2 = (100 * 0.2 + 50 * 0) / 100
    carbs2 = (100 * 76.3 + 50 * 100) / 100
    sugar2 = (100 * 0.3 + 50 * 100) / 100
    free_sugar2 = (100 * 0 + 50 * 100) / 100
    fibres2 = (100 * 2.7 + 50 * 0) / 100
    sel2 = (100 * 0.01 + 50 * 0) / 100
    alcool2 = (100 * 0 + 50 * 0) / 100
    water2 = (100 * 12.0 + 50 * 0) / 100
    sport2 = 0

    # Row 3
    weight3 = 80.1
    calories3 = (10 * 40 + 0.682 * (259 * 0.8 + 239 * 0.2)) / 100
    alcool3 = (10 * 0.5 + 0.682 * (0 * 0.8 + 0 * 0.2)) / 100
    sport3 = WALKING_CALORIES(
        duration_minutes=3600, steps=5, incline_percent=0, weight_kg=weight3
    )

    # Row 4
    (
        calories4,
        proteine4,
        fat4,
        sfat4,
        carbs4,
        sugar4,
        free_sugar4,
        fibres4,
        sel4,
        alcool4,
        water4,
    ) = [0] * 11
    sport4 = 10 * 10

    # Assert calculated values are close to expected values
    expected_values = {
        "2024-01-01": [
            80.0,
            calories1,
            proteine1,
            fat1,
            sfat1,
            carbs1,
            sugar1,
            free_sugar1,
            fibres1,
            sel1,
            alcool1,
            water1,
            sport1,
        ],
        "2024-01-02": [
            80.2,
            calories2,
            proteine2,
            fat2,
            sfat2,
            carbs2,
            sugar2,
            free_sugar2,
            fibres2,
            sel2,
            alcool2,
            water2,
            sport2,
        ],
        "2024-01-03": [
            80.1,
            calories3,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            alcool3,
            np.nan,
            sport3,
        ],  # Many NaNs because formula is incomplete
        "2024-01-04": [
            79.9,
            calories4,
            proteine4,
            fat4,
            sfat4,
            carbs4,
            sugar4,
            free_sugar4,
            fibres4,
            sel4,
            alcool4,
            water4,
            sport4,
        ],
    }

    for date, values in expected_values.items():
        for col, expected_val in zip(expected_columns, values):
            actual_val = result_df.loc[date, col]
            if pd.isna(expected_val):
                assert pd.isna(actual_val), (
                    f"Column '{col}' for date '{date}' should be NaN"
                )
            else:
                assert np.isclose(actual_val, expected_val), (
                    f"Column '{col}' for date '{date}' failed: expected {expected_val}, got {actual_val}"
                )

    # Verify that the to_csv mock was called correctly
    mock_to_csv.assert_called_once_with("data/features.csv")
