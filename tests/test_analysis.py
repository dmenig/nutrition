import unittest
import os
import pandas as pd
import json
from analyze_results import analyze_results


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up a dummy environment for testing."""
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists("plots"):
            os.makedirs("plots")

        # Create dummy data and artifacts
        self.features_df = pd.DataFrame(
            {
                "calories": [2000, 2200, 2100],
                "sport": [300, 400, 350],
                "pds": [70, 70.2, 70.1],
                "carbs": [100, 150, 120],
                "sugar": [20, 30, 25],
                "sel": [1000, 1200, 1100],
                "alcool": [0, 1, 0],
                "water": [2000, 2500, 2200],
            }
        )
        self.features_df.to_csv("data/features.csv", index=False)

        self.best_params = {
            "initial_M_base": 2000,
            "alpha": 0.1,
            "look_back_window": 7,
            "K_cal_kg": 7700,
        }
        with open("best_params.json", "w") as f:
            json.dump(self.best_params, f)

        self.loss_weights = {"w_meta": 1.0, "w_water": 1.0}
        with open("loss_weights.json", "w") as f:
            json.dump(self.loss_weights, f)

    def test_analyze_results_runs_without_errors(self):
        """Test that analyze_results.py runs without errors and the sanity check passes."""
        try:
            analyze_results()
            # Check that plots are created
            self.assertTrue(os.path.exists("plots/weight_overview.png"))
            self.assertTrue(os.path.exists("plots/water_retention.png"))
            self.assertTrue(os.path.exists("plots/metabolism.png"))
        except Exception as e:
            self.fail(f"analyze_results() raised an exception: {e}")

    def tearDown(self):
        """Clean up dummy files and directories."""
        if os.path.exists("data/features.csv"):
            os.remove("data/features.csv")
        if os.path.exists("best_params.json"):
            os.remove("best_params.json")
        if os.path.exists("loss_weights.json"):
            os.remove("loss_weights.json")
        if os.path.exists("plots/weight_overview.png"):
            os.remove("plots/weight_overview.png")
        if os.path.exists("plots/water_retention.png"):
            os.remove("plots/water_retention.png")
        if os.path.exists("plots/metabolism.png"):
            os.remove("plots/metabolism.png")
        if os.path.exists("data/features.csv"):
            os.remove("data/features.csv")
        if os.path.exists("data"):
            if not os.listdir("data"):
                os.rmdir("data")
        if os.path.exists("plots"):
            if not os.listdir("plots"):
                os.rmdir("plots")


if __name__ == "__main__":
    unittest.main()
