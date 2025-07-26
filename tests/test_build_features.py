import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, call

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from build_features import main as build_features_main
from sport_formulas import WALKING_CALORIES, RUNNING_CALORIES
from data_processor import normalize_food_names, save_normalized_variables


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
                "caviar d'aubergine",
            ],
            "Calories / 100g": [364, 387, 259, 364, 239, 40, 150],
            "Protéine": [10.3, 0, 13.4, 21.6, 27.0, 0, 2.0],
            "Fat": [1.0, 0, 3.2, 29.8, 14.0, 0, 12.0],
            "SFat": [0.2, 0, 0.7, 20.2, 3.8, 0, 1.5],
            "Carbs": [76.3, 100, 41.3, 1.3, 0, 10.6, 5.0],
            "Sugar": [0.3, 100, 5.4, 0.5, 0, 10.6, 3.0],
            "Free sugar": [0, 100, 1.0, 0, 0, 10.6, 0.5],
            "Fibres": [2.7, 0, 6.0, 0, 0, 0, 4.0],
            "Sel": [0.01, 0, 1.4, 1.8, 0.2, 0.01, 1.2],
            "Alcool": [0, 0, 0, 0, 0, 0.5, 0],
            "Water": [12.0, 0, 35.0, 45.0, 60.0, 88.0, 75.0],
        }
    )

    journal_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "Pds": [80.0, 80.2, 80.1, 79.9, 79.8],
            "Nourriture": [
                "150 * (pain_complet * 0.5 + fromage_de_chevre * 0.5) + 200 * poulet",
                "100 * farine + 50 * sucre",
                "10 * seven_up_mojito + 0.682* (pain_complet*0.8 + poulet*0.2)",
                "",  # Test empty food formula
                "100 * caviar_d_aubergine",  # Test normalized name
            ],
            "Sport": [
                "15*8+ 2*WALKING_CALORIES(duration_minutes=750, distance_meters=10000, additional_weight_kg=0, weight_kg=WEIGHT) + RUNNING_CALORIES(duration_minutes=1200, distance_meters=10000, additional_weight_kg=0, weight_kg=WEIGHT)",
                "",
                "WALKING_CALORIES(duration_minutes=3600, distance_meters=15000, additional_weight_kg=0, weight_kg=WEIGHT)",
                "10*10",  # Test simple sport formula
                "",
            ],
        }
    )

    # Create the expected normalized dataframe
    normalized_variables_df = variables_df.copy()
    normalized_variables_df["Nom"] = normalized_variables_df["Nom"].apply(
        normalize_food_names
    )

    # 2. Use `patch` to mock `pd.read_csv` and `df.to_csv`
    with (
        patch("pandas.read_csv") as mock_read_csv,
        patch("pandas.DataFrame.to_csv") as mock_to_csv,
        patch("os.path.exists") as mock_exists,
    ):
        # Mock os.path.exists to return True, so the script thinks
        # normalized_variables.csv already exists.
        mock_exists.return_value = True

        # Configure the mock to return different dataframes based on the file path
        def read_csv_side_effect(filepath):
            if "journal" in filepath:
                return journal_df.copy()
            elif "normalized_variables" in filepath:
                # The script should now be reading the normalized file
                return normalized_variables_df.copy()
            elif "variables.csv" in filepath:
                # This case is for completeness, though it won't be hit in this setup
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
    )
    proteine1 = (
        150 * (13.4 * pain_complet_ratio + 21.6 * fromage_chevre_ratio) + 200 * 27.0
    )
    fat1 = (
        150 * (3.2 * pain_complet_ratio + 29.8 * fromage_chevre_ratio) + 200 * 14.0
    )
    sfat1 = (
        150 * (0.7 * pain_complet_ratio + 20.2 * fromage_chevre_ratio) + 200 * 3.8
    )
    carbs1 = (
        150 * (41.3 * pain_complet_ratio + 1.3 * fromage_chevre_ratio) + 200 * 0
    )
    sugar1 = (
        150 * (5.4 * pain_complet_ratio + 0.5 * fromage_chevre_ratio) + 200 * 0
    )
    free_sugar1 = (
        150 * (1.0 * pain_complet_ratio + 0 * fromage_chevre_ratio) + 200 * 0
    )
    fibres1 = (
        150 * (6.0 * pain_complet_ratio + 0 * fromage_chevre_ratio) + 200 * 0
    )
    sel1 = (
        150 * (1.4 * pain_complet_ratio + 1.8 * fromage_chevre_ratio) + 200 * 0.2
    )
    alcool1 = (
        150 * (0 * pain_complet_ratio + 0 * fromage_chevre_ratio) + 200 * 0
    )
    water1 = (
        150 * (35.0 * pain_complet_ratio + 45.0 * fromage_chevre_ratio) + 200 * 60.0
    )
    sport1 = (
        15 * 8
        + 2
        * WALKING_CALORIES(
            duration_minutes=750,
            distance_meters=10000,
            additional_weight_kg=0,
            weight_kg=weight1,
        )
        + RUNNING_CALORIES(
            duration_minutes=1200,
            distance_meters=10000,
            additional_weight_kg=0,
            weight_kg=weight1,
        )
    )

    # Row 2
    calories2 = (100 * 364 + 50 * 387)
    proteine2 = (100 * 10.3 + 50 * 0)
    fat2 = (100 * 1.0 + 50 * 0)
    sfat2 = (100 * 0.2 + 50 * 0)
    carbs2 = (100 * 76.3 + 50 * 100)
    sugar2 = (100 * 0.3 + 50 * 100)
    free_sugar2 = (100 * 0 + 50 * 100)
    fibres2 = (100 * 2.7 + 50 * 0)
    sel2 = (100 * 0.01 + 50 * 0)
    alcool2 = (100 * 0 + 50 * 0)
    water2 = (100 * 12.0 + 50 * 0)
    sport2 = 0

    # Row 3
    weight3 = 80.1
    calories3 = (10 * 40 + 0.682 * (259 * 0.8 + 239 * 0.2))
    proteine3 = (10 * 0 + 0.682 * (13.4 * 0.8 + 27.0 * 0.2))
    fat3 = (10 * 0 + 0.682 * (3.2 * 0.8 + 14.0 * 0.2))
    sfat3 = (10 * 0 + 0.682 * (0.7 * 0.8 + 3.8 * 0.2))
    carbs3 = (10 * 10.6 + 0.682 * (41.3 * 0.8 + 0 * 0.2))
    sugar3 = (10 * 10.6 + 0.682 * (5.4 * 0.8 + 0 * 0.2))
    free_sugar3 = (10 * 10.6 + 0.682 * (1.0 * 0.8 + 0 * 0.2))
    fibres3 = (10 * 0 + 0.682 * (6.0 * 0.8 + 0 * 0.2))
    sel3 = (10 * 0.01 + 0.682 * (1.4 * 0.8 + 0.2 * 0.2))
    alcool3 = (10 * 0.5 + 0.682 * (0 * 0.8 + 0 * 0.2))
    water3 = (10 * 88.0 + 0.682 * (35.0 * 0.8 + 60.0 * 0.2))
    sport3 = WALKING_CALORIES(
        duration_minutes=3600,
        distance_meters=15000,
        additional_weight_kg=0,
        weight_kg=weight3,
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

    # Row 5
    calories5 = (100 * 150)
    proteine5 = (100 * 2.0)
    fat5 = (100 * 12.0)
    sfat5 = (100 * 1.5)
    carbs5 = (100 * 5.0)
    sugar5 = (100 * 3.0)
    free_sugar5 = (100 * 0.5)
    fibres5 = (100 * 4.0)
    sel5 = (100 * 1.2)
    alcool5 = (100 * 0)
    water5 = (100 * 75.0)
    sport5 = 0

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
            proteine3,
            fat3,
            sfat3,
            carbs3,
            sugar3,
            free_sugar3,
            fibres3,
            sel3,
            alcool3,
            water3,
            sport3,
        ],  # Formula is complete
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
        "2024-01-05": [
            79.8,
            calories5,
            proteine5,
            fat5,
            sfat5,
            carbs5,
            sugar5,
            free_sugar5,
            fibres5,
            sel5,
            alcool5,
            water5,
            sport5,
        ],
    }

    for date, values in expected_values.items():
        date_ts = pd.Timestamp(date)  # Convert string to Timestamp
        for col, expected_val in zip(expected_columns, values):
            actual_val = result_df.loc[date_ts, col]
            if pd.isna(expected_val):
                assert pd.isna(actual_val), (
                    f"Column '{col}' for date '{date}' should be NaN"
                )
            else:
                assert np.isclose(actual_val, expected_val), (
                    f"Column '{col}' for date '{date}' failed: expected {expected_val}, got {actual_val}"
                )

    # Verify that the to_csv mock was called correctly
    expected_calls = [
        call("data/normalized_variables.csv", index=False),
        call("data/features.csv"),
    ]
    mock_to_csv.assert_has_calls(expected_calls, any_order=False)
