import unittest
import pandas as pd
import optuna
from unittest.mock import MagicMock, patch

from objective import objective, run_model_simulation
from weight_model import WeightModel


class TestObjective(unittest.TestCase):
    def setUp(self):
        """Set up a dummy features DataFrame for testing."""
        self.features_df = pd.DataFrame(
            {
                "C_in": [2500, 2600, 2400, 2700, 2550],
                "W_act": [80.0, 80.2, 80.1, 80.3, 80.4],
                "carbs": [300, 320, 280, 330, 310],
                "sugar": [80, 90, 70, 95, 85],
                "sel": [4.5, 5.0, 4.0, 5.5, 4.8],
                "alcool": [0, 0, 1, 0, 0],
                "water": [2.0, 2.2, 1.8, 2.5, 2.1],
                "sport": [300, 400, 200, 500, 350],
                "weight": [80.0, 80.2, 80.1, 80.3, 80.4],
                "calories": [2500, 2600, 2400, 2700, 2550],
            }
        )
        self.w_meta = 1.0
        self.w_water = 1.0
        self.f_water_model_params = {"model_type": "LinearRegression"}

    def test_objective_returns_float(self):
        """Test that the objective function returns a float."""
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(
                trial,
                self.features_df,
                self.w_meta,
                self.w_water,
                self.f_water_model_params,
            ),
            n_trials=1,
        )
        self.assertIsInstance(study.best_value, float)

    @patch("objective.WeightModel")
    def test_two_pass_training_logic(self, mock_weight_model):
        """Test that the WaterRetentionModel is trained in the second pass."""
        mock_instance = MagicMock()
        mock_weight_model.return_value = mock_instance

        # Mock the run_simulation to return a dummy DataFrame
        dummy_results = pd.DataFrame(
            {"M_base": [2000], "WR": [0], "W_act_t": [80.0], "W_obs_t": [80.0]}
        )
        mock_instance.run.return_value = dummy_results

        params = {
            "initial_M_base": 2000,
            "alpha": 0.1,
            "look_back_window": 7,
            "K_cal_kg": 7700,
        }

        run_model_simulation(params, self.features_df, self.f_water_model_params)

        # Check that fit_water_model was called once
        mock_instance.fit_water_model.assert_called_once()

        # Check that run was called twice
        self.assertEqual(mock_instance.run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
