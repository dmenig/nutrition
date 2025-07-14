import os
import sys
import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import build_features
from data_processor import load_and_process_data as original_load_and_process_data


@pytest.fixture
def dummy_data_paths(tmpdir):
    """Creates dummy CSV files and returns their paths."""
    data_dir = tmpdir.mkdir("data")

    journal_path = data_dir.join("processed_journal.csv")
    variables_path = data_dir.join("variables.csv")

    # Dummy journal data
    journal_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-07-01", "2024-07-02"]),
            "Pds": [70.0, 70.5],
            "Nourriture": ["100 * pain", "50 * fromage"],
            "Sport": ["", ""],
        }
    )
    journal_df.to_csv(journal_path, index=False)

    # Dummy variables data
    variables_df = pd.DataFrame(
        {
            "Nom": ["pain", "fromage"],
            "Calories / 100g": [250, 400],
            "Proteines / 100g": [10, 25],
        }
    )
    variables_df.to_csv(variables_path, index=False)

    return str(journal_path), str(variables_path)


def test_build_features_main(dummy_data_paths, monkeypatch):
    """
    Tests the main script execution of build_features.py.
    """
    journal_path, variables_path = dummy_data_paths

    # We patch 'load_and_process_data' to use our dummy files.
    def mock_load_and_process_data():
        return original_load_and_process_data(
            journal_path=journal_path, variables_path=variables_path
        )

    monkeypatch.setattr(
        build_features.data_processor,
        "load_and_process_data",
        mock_load_and_process_data,
    )

    # Run the main script function
    result_df = build_features.main()

    # Assertions
    assert not result_df.empty
    assert "pds" in result_df.columns
    assert "calories___100g" in result_df.columns
    assert "proteines___100g" in result_df.columns
    assert len(result_df) == 2
    # Check a calculated value
    # For day 1: 100 * pain -> 1 * 250 = 250 calories
    assert result_df.loc[pd.to_datetime("2024-07-01")]["calories___100g"] == 250.0
    # For day 2: 50 * fromage -> 0.5 * 400 = 200 calories
    assert result_df.loc[pd.to_datetime("2024-07-02")]["calories___100g"] == 200.0
