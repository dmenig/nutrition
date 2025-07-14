import numpy as np
import pandas as pd
import pytest
from weight_model import WaterRetentionModel, WeightModel


def test_water_retention_model():
    """
    Tests the fit and predict methods of the WaterRetentionModel.
    """
    model = WaterRetentionModel()
    X = pd.DataFrame({"a": np.random.rand(10), "b": np.random.rand(10)})
    y = pd.Series(np.random.rand(10))
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (10,)


def test_prepare_f_water_features():
    """
    Tests the feature preparation logic.
    """
    weight_model = WeightModel(
        K_cal_kg=7700,
        initial_M_base=2000,
        alpha=0.1,
        f_water_model_params={},
        look_back_window=5,
    )
    data_df = pd.DataFrame(
        {
            "carbs": [10, 20, 30, 40, 50, 60],
            "sugar": [1, 2, 3, 4, 5, 6],
            "sel": [1, 1, 1, 1, 1, 1],
            "alcool": [0, 0, 0, 0, 0, 0],
            "water": [2, 2, 2, 2, 2, 2],
            "sport": [100, 100, 100, 100, 100, 100],
        }
    )
    results_df = pd.DataFrame(
        {
            "C_in_t": [2000, 2100, 2200, 2300, 2400, 2500],
            "C_exp_t": [2100, 2100, 2100, 2100, 2100, 2100],
            "W_act_t": [70, 69.9, 69.8, 69.7, 69.6, 69.5],
        }
    )
    features = weight_model._prepare_f_water_features(5, data_df, results_df)
    assert isinstance(features, pd.DataFrame)
    assert not features.isnull().values.any()
    expected_cols = [
        "carbs_mean",
        "carbs_std",
        "sugar_mean",
        "sugar_std",
        "sel_mean",
        "sel_std",
        "alcool_mean",
        "alcool_std",
        "water_mean",
        "water_std",
        "sport_mean",
        "sport_std",
        "c_in_minus_c_exp",
        "w_act_t_minus_1",
    ]
    assert all(col in features.columns for col in expected_cols)


def test_weight_model_run():
    """
    Tests the run method with a small, controlled dataset.
    """
    data_df = pd.DataFrame(
        {
            "weight": [70, 70.1, 70.2, 70.3, 70.4, 70.5, 70.6],
            "calories": [2500, 2600, 2400, 2700, 2550, 2650, 2450],
            "sport": [300, 400, 200, 500, 350, 450, 250],
            "carbs": [300, 320, 280, 350, 310, 330, 290],
            "sugar": [50, 55, 45, 60, 52, 58, 48],
            "sel": [4, 4.2, 3.8, 4.5, 4.1, 4.3, 3.9],
            "alcool": [0, 0, 0, 0, 0, 0, 0],
            "water": [2, 2.1, 1.9, 2.2, 2.0, 2.1, 1.9],
        }
    )

    model = WeightModel(
        K_cal_kg=7700,
        initial_M_base=2000,
        alpha=0.1,
        f_water_model_params={"model_type": "LinearRegression"},
        look_back_window=5,
    )

    initial_results_df = model.run(data_df)
    model.fit_water_model(data_df, initial_results_df)
    results_df = model.run(data_df)
    assert isinstance(results_df, pd.DataFrame)
    assert len(results_df) == len(data_df)
    expected_cols = [
        "t",
        "C_in_t",
        "C_exp_t",
        "M_t",
        "W_act_t",
        "WR_t",
        "W_pred_t",
        "W_obs_t",
    ]
    assert all(col in results_df.columns for col in expected_cols)
