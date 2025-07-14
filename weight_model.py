import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class WaterRetentionModel:
    """
    A wrapper around a regression model and a StandardScaler.
    """

    def __init__(self, model_type: str = "LinearRegression", **kwargs):
        """
        Initializes the scaler and the specified regression model.
        """
        self.scaler = StandardScaler()
        if model_type == "LinearRegression":
            self.model = LinearRegression(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Scales the features X and fits the model.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Scales the input features X using the already-fitted scaler
        and returns the model's predictions.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class WeightModel:
    """
    Performs the main simulation, iterating through data day by day.
    """

    def __init__(
        self,
        K_cal_kg: float,
        initial_M_base: float,
        alpha: float,
        f_water_model_params: dict,
        look_back_window: int,
    ):
        self.K_cal_kg = K_cal_kg
        self.initial_M_base = initial_M_base
        self.alpha = alpha
        self.f_water_model = WaterRetentionModel(**f_water_model_params)
        self.look_back_window = look_back_window

    def _prepare_f_water_features(
        self, current_idx: int, data_df: pd.DataFrame, results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Constructs features for the WaterRetentionModel.
        """
        if current_idx < 1:
            return None

        start_idx = max(0, current_idx - self.look_back_window)
        window_data = data_df.iloc[start_idx:current_idx]

        features = {}

        input_features = ["carbs", "sugar", "sel", "alcool", "water", "sport"]

        for feature in input_features:
            features[f"{feature}_mean"] = window_data[feature].mean()
            features[f"{feature}_std"] = window_data[feature].std()

        prev_result = results_df.iloc[current_idx - 1]
        c_in_t_minus_1 = prev_result["C_in_t"]
        c_exp_t_minus_1 = prev_result["C_exp_t"]
        w_act_t_minus_1 = prev_result["W_act_t"]

        features["c_in_minus_c_exp"] = c_in_t_minus_1 - c_exp_t_minus_1
        features["w_act_t_minus_1"] = w_act_t_minus_1

        return pd.DataFrame([features]).fillna(0)

    def fit_water_model(self, data_df: pd.DataFrame, results_df: pd.DataFrame):
        """
        Trains the water retention model using historical data.
        """
        historical_X_list = []
        historical_y_list = []
        for i in range(1, len(results_df)):
            features = self._prepare_f_water_features(i, data_df, results_df)
            if features is not None:
                historical_X_list.append(features)
                w_obs_i = results_df.loc[i, "W_obs_t"]
                w_act_i = results_df.loc[i, "W_act_t"]
                historical_y_list.append(w_obs_i - w_act_i)

        if len(historical_X_list) > 1:
            historical_X = pd.concat(historical_X_list, ignore_index=True)
            historical_y = pd.Series(historical_y_list)
            self.f_water_model.fit(historical_X, historical_y)

    def run(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        The core simulation loop.
        """
        results_list = []
        M_t = self.initial_M_base

        W_act_t = data_df.loc[0, "weight"]
        C_in_t = data_df.loc[0, "calories"]
        C_exp_t = M_t + data_df.loc[0, "sport"]

        results_list.append(
            {
                "t": 0,
                "C_in_t": C_in_t,
                "C_exp_t": C_exp_t,
                "M_t": M_t,
                "W_act_t": W_act_t,
                "WR_t": 0,
                "W_pred_t": W_act_t,
                "W_obs_t": data_df.loc[0, "weight"],
            }
        )

        for t in range(1, len(data_df)):
            prev_result = results_list[t - 1]
            C_in_t_minus_1 = prev_result["C_in_t"]
            C_exp_t_minus_1 = prev_result["C_exp_t"]
            M_t_minus_1 = prev_result["M_t"]
            W_act_t_minus_1 = prev_result["W_act_t"]

            W_obs_t = data_df.loc[t, "weight"]
            W_obs_t_minus_1 = data_df.loc[t - 1, "weight"]

            W_act_t = (
                W_act_t_minus_1 + (C_in_t_minus_1 - C_exp_t_minus_1) / self.K_cal_kg
            )
            M_t = (
                self.alpha
                * (
                    C_in_t_minus_1
                    - C_exp_t_minus_1
                    - (W_obs_t - W_obs_t_minus_1) * self.K_cal_kg
                )
                + (1 - self.alpha) * M_t_minus_1
            )
            C_in_t = data_df.loc[t, "calories"]
            C_exp_t = M_t + data_df.loc[t, "sport"]

            results_df = pd.DataFrame(results_list)
            X_t = self._prepare_f_water_features(t, data_df, results_df)

            WR_t = 0
            if X_t is not None and self.f_water_model.is_fitted:
                WR_t = self.f_water_model.predict(X_t)[0]

            W_pred_t = W_act_t + WR_t

            results_list.append(
                {
                    "t": t,
                    "C_in_t": C_in_t,
                    "C_exp_t": C_exp_t,
                    "M_t": M_t,
                    "W_act_t": W_act_t,
                    "WR_t": WR_t,
                    "W_pred_t": W_pred_t,
                    "W_obs_t": W_obs_t,
                }
            )

        return pd.DataFrame(results_list)
