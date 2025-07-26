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
        self,
        current_idx: int,
        data_df: pd.DataFrame,
        results_df: pd.DataFrame,
        w_act_col: str = "W_act_t",
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
        w_act_t_minus_1 = prev_result[w_act_col]

        features["c_in_minus_c_exp"] = c_in_t_minus_1 - c_exp_t_minus_1
        features["w_act_t_minus_1"] = w_act_t_minus_1

        return pd.DataFrame([features]).fillna(0)

    def fit_water_model(self, data_df: pd.DataFrame, results_df: pd.DataFrame):
        """
        Trains the water retention model using historical data.
        """
        historical_X_list = []
        historical_y_list = []
        w_act_col = "W_act" if "W_act" in results_df.columns else "W_act_t"
        for i in range(1, len(results_df)):
            features = self._prepare_f_water_features(
                i, data_df, results_df, w_act_col=w_act_col
            )
            if features is not None:
                historical_X_list.append(features)
                w_obs_i = results_df.loc[i, "W_obs_t"]
                w_act_i = results_df.loc[i, w_act_col]
                historical_y_list.append(w_obs_i - w_act_i)

        if len(historical_X_list) > 1:
            historical_X = pd.concat(historical_X_list, ignore_index=True)
            historical_y = pd.Series(historical_y_list)

            # Drop NaN values
            valid_indices = ~historical_y.isna()
            historical_X = historical_X[valid_indices]
            historical_y = historical_y[valid_indices]

            if not historical_X.empty:
                self.f_water_model.fit(historical_X, historical_y)

    def run(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        The core simulation loop.
        """
        num_days = len(data_df)
        results = np.zeros(
            num_days,
            dtype=[
                ("t", "i4"),
                ("C_in_t", "f8"),
                ("C_exp_t", "f8"),
                ("M_base", "f8"),
                ("W_act", "f8"),
                ("WR_t", "f8"),
                ("W_pred_t", "f8"),
                ("W_obs_t", "f8"),
            ],
        )

        # Initialize base arrays
        calories_in = data_df["calories"].values
        sport_exp = data_df["sport"].values
        w_obs = data_df["pds"].values

        # Set initial values
        results["t"] = np.arange(num_days)
        results["W_obs_t"] = w_obs
        results["W_act"][0] = w_obs[0]
        results["M_base"][0] = self.initial_M_base
        results["C_in_t"][0] = calories_in[0]
        results["C_exp_t"][0] = results["M_base"][0] + sport_exp[0]
        results["W_pred_t"][0] = results["W_act"][0]

        # Iterative calculation of metabolism and weight
        for t in range(1, num_days):
            # Correctly calculate M_base for day t using rearranged formula
            m_base_t_minus_1 = results["M_base"][t - 1]
            c_in_t = calories_in[t]
            sport_exp_t = sport_exp[t]
            w_obs_t = w_obs[t]
            w_obs_t_minus_1 = w_obs[t - 1]

            numerator = (
                self.alpha
                * (c_in_t - sport_exp_t - (w_obs_t - w_obs_t_minus_1) * self.K_cal_kg)
                + (1 - self.alpha) * m_base_t_minus_1
            )
            denominator = 1 + self.alpha
            results["M_base"][t] = numerator / denominator

            # Update energy expenditure for day t
            results["C_in_t"][t] = c_in_t
            results["C_exp_t"][t] = results["M_base"][t] + sport_exp[t]

            # Update actual weight for day t based on previous day's balance
            calorie_delta_t_minus_1 = (
                results["C_in_t"][t - 1] - results["C_exp_t"][t - 1]
            )
            results["W_act"][t] = (
                results["W_act"][t - 1] + calorie_delta_t_minus_1 / self.K_cal_kg
            )

        # Water retention and final prediction
        if self.f_water_model.is_fitted:
            temp_results_df = pd.DataFrame(results)
            for t in range(1, num_days):
                X_t = self._prepare_f_water_features(
                    t, data_df, temp_results_df, w_act_col="W_act"
                )
                if X_t is not None:
                    results["WR_t"][t] = self.f_water_model.predict(X_t)[0]

        results["W_pred_t"] = results["W_act"] + results["WR_t"]
        return pd.DataFrame(results)
