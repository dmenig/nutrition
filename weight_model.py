import pandas as pd
import numpy as np
from water_retention_model import WaterRetentionModel


class WeightModel:
    def __init__(
        self,
        K_cal_kg,
        initial_M_base,
        alpha,
        f_water_model_params=None,
        look_back_window=7,
    ):
        """
        Initializes the WeightModel.

        Args:
            K_cal_kg (float): Energy equivalent of 1kg of body weight (e.g., 7700 kcal/kg).
            initial_M_base (float): Initial estimated base metabolic rate in kcal/day.
            alpha (float): Smoothing factor for Exponential Moving Average (EMA) of metabolism.
                           Value between 0 and 1. Higher alpha means faster adaptation.
            f_water_model_params (dict): Dictionary of parameters for initializing WaterRetentionModel.
                                         If None, default linear model will be used.
            look_back_window (int): Number of past days to consider for f_water_model features.
        """
        self.K_cal_kg = K_cal_kg
        self.initial_M_base = initial_M_base
        self.alpha = alpha
        self.f_water_model_params = (
            f_water_model_params
            if f_water_model_params is not None
            else {"model_type": "linear"}
        )
        self.f_water_model = WaterRetentionModel(**self.f_water_model_params)
        self.look_back_window = look_back_window

        # Store historical states for running the model
        self.history = {
            "W_act": [],
            "M_base": [],
            "WR": [],
            "W_obs": [],
            "C_in": [],
            "C_exp_activity": [],
            "Nutritional_Vector": [],  # Store full nutritional vectors for f_water
        }

    def _prepare_f_water_features(self, current_day_index, features_df):
        """
        Prepares features for the f_water model based on the look-back window.
        """
        # Features for f_water:
        # 1. Observed_Difference (calculated in run method)
        # 2. Current Nutritional Vector
        # 3. Historical Nutritional Vectors (from look_back_window)
        # 4. Historical Observed Weights (from look_back_window)

        # Ensure we have enough data for the look-back window
        if current_day_index < self.look_back_window:
            return None  # Not enough history to form features

        # Get the window of historical data
        start_index = current_day_index - self.look_back_window
        window_df = features_df.iloc[start_index:current_day_index]

        # Extract nutritional columns (excluding 'Pds' and 'Sport')
        nutritional_cols = [
            col for col in features_df.columns if col not in ["Pds", "Sport"]
        ]

        # Flatten historical nutritional vectors
        historical_nutritional_features = window_df[nutritional_cols].values.flatten()

        # Flatten historical observed weights
        historical_pds_features = window_df["Pds"].values.flatten()

        # Combine all features. Observed_Difference will be added in the run method.
        # The current day's nutritional vector will also be added in the run method.

        # The f_water model will be trained to predict 'p' based on:
        # [Observed_Difference_t, current_nutritional_vector, historical_nutritional_features, historical_pds_features]

        # For now, this method just prepares the historical part.
        # The current day's features and Observed_Difference will be added right before prediction.

        return {
            "historical_nutritional": historical_nutritional_features,
            "historical_pds": historical_pds_features,
        }

    def run(self, data_df):
        """
        Runs the weight model over the provided time-series data.

        Args:
            data_df (pd.DataFrame): DataFrame containing daily data with columns
                                    'Pds' (observed weight), 'Sport' (activity calories),
                                    and all nutritional columns (e.g., 'calories', 'proteines', etc.).

        Returns:
            dict: A dictionary containing time-series results for 'W_act', 'WR', 'M_base'.
        """
        self.history = {
            "W_act": [
                self.initial_M_base / 20
            ],  # Initial W_act is arbitrary, will be tuned
            "M_base": [self.initial_M_base],
            "WR": [0.0],  # Initial water retention
            "W_obs": [],
            "C_in": [],
            "C_exp_activity": [],
            "Nutritional_Vector": [],
        }

        # Extract nutritional columns for C_in and the full vector
        nutritional_cols = [
            col for col in data_df.columns if col not in ["Pds", "Sport"]
        ]

        # Ensure 'calories' is present for C_in
        if "calories" not in nutritional_cols:
            raise ValueError(
                "DataFrame must contain a 'calories' column for C_in calculation."
            )

        for i in range(len(data_df)):
            current_date_data = data_df.iloc[i]
            W_obs_t = current_date_data["Pds"]
            C_exp_activity_t = current_date_data[
                "sport"
            ]  # Assuming 'sport' column contains activity calories
            C_in_t = current_date_data["calories"]

            # Extract full nutritional vector for the current day
            nutritional_vector_t = current_date_data[nutritional_cols].values.tolist()

            # 1. Predict Weight Change from Calories
            M_base_prev = self.history["M_base"][-1]
            W_act_prev = self.history["W_act"][-1]

            C_total_exp_t = M_base_prev + C_exp_activity_t
            C_delta_t = C_in_t - C_total_exp_t
            W_act_predicted_from_calories_t = W_act_prev + C_delta_t / self.K_cal_kg

            # 2. Calculate Observed Difference
            Observed_Difference_t = W_obs_t - W_act_predicted_from_calories_t

            # 3. Attribute Difference to Water vs. Metabolism using f_water
            # The f_water model predicts the proportion 'p' of Observed_Difference_t
            # that is attributable to water retention.

            if i < self.look_back_window:
                # Not enough history for f_water, assume a default split for initial days
                # This can be a tunable parameter or a fixed value.
                # For now, let's assume a default of 0.5, meaning half goes to WR, half to W_act correction.
                p_t = 0.5
            else:
                # Prepare features for f_water_model
                historical_features = self._prepare_f_water_features(i, data_df)

                # Combine current day's features with historical features
                # The order of features must be consistent with how the f_water_model is trained.
                # Let's define the feature vector for f_water_model as:
                # [Observed_Difference_t, current_nutritional_vector, historical_nutritional_features, historical_pds_features]

                X_f_water_list = (
                    [Observed_Difference_t]
                    + nutritional_vector_t
                    + historical_features["historical_nutritional"].tolist()
                    + historical_features["historical_pds"].tolist()
                )

                X_f_water = np.array(X_f_water_list).reshape(1, -1)

                # Predict the proportion 'p' using the f_water_model
                p_t = self.f_water_model.predict(X_f_water)[0]

            WR_t = p_t * Observed_Difference_t
            W_act_correction_t = (1 - p_t) * Observed_Difference_t
            W_act_t = W_act_predicted_from_calories_t + W_act_correction_t

            # 5. Update Base Metabolism with EMA Smoothing
            C_discrepancy_t = W_act_correction_t * self.K_cal_kg
            M_base_target_t = M_base_prev - C_discrepancy_t
            M_base_t = (1 - self.alpha) * M_base_prev + self.alpha * M_base_target_t

            # Store results
            self.history["W_act"].append(W_act_t)
            self.history["M_base"].append(M_base_t)
            self.history["WR"].append(WR_t)
            self.history["W_obs"].append(W_obs_t)
            self.history["C_in"].append(C_in_t)
            self.history["C_exp_activity"].append(C_exp_activity_t)
            self.history["Nutritional_Vector"].append(nutritional_vector_t)

        # Convert lists to pandas Series for easier handling
        results = {
            "W_act": pd.Series(self.history["W_act"][1:], index=data_df.index),
            "M_base": pd.Series(self.history["M_base"][1:], index=data_df.index),
            "WR": pd.Series(self.history["WR"], index=data_df.index),
            "W_obs": pd.Series(self.history["W_obs"], index=data_df.index),
            "C_in": pd.Series(self.history["C_in"], index=data_df.index),
            "C_exp_activity": pd.Series(
                self.history["C_exp_activity"], index=data_df.index
            ),
        }
        return results


if __name__ == "__main__":
    # Example usage with dummy data
    print("Running dummy example for WeightModel with EMA...")

    # Create dummy data_df similar to what data_processor.py would output
    dummy_data = {
        "Date": pd.to_datetime(
            [
                "2024-07-01",
                "2024-07-02",
                "2024-07-03",
                "2024-07-04",
                "2024-07-05",
                "2024-07-06",
                "2024-07-07",
                "2024-07-08",
                "2024-07-09",
                "2024-07-10",
            ]
        ),
        "Pds": [70.0, 70.5, 70.2, 70.8, 70.3, 70.6, 70.1, 70.4, 70.0, 70.5],
        "sport": [300, 200, 0, 400, 100, 0, 250, 150, 0, 300],  # Activity calories
        "calories": [2000, 2200, 1800, 2500, 1900, 2100, 2300, 2000, 1700, 2400],
        "proteines": [100, 110, 90, 120, 95, 105, 115, 100, 85, 120],
        "glucides": [200, 250, 180, 300, 190, 220, 260, 200, 170, 280],
        "lipides": [70, 80, 60, 90, 65, 75, 85, 70, 55, 80],
        "alcool": [0, 100, 0, 0, 50, 0, 0, 75, 0, 0],
    }
    dummy_df = pd.DataFrame(dummy_data).set_index("Date")

    # Initialize model with dummy values
    K_cal_kg = 7700
    initial_M_base = 2000  # kcal/day
    alpha = 0.1  # Smoothing factor for EMA

    # Initialize WeightModel with default WaterRetentionModel (LinearRegression)
    model = WeightModel(
        K_cal_kg=K_cal_kg,
        initial_M_base=initial_M_base,
        alpha=alpha,
        look_back_window=3,
    )

    # To run the model, we need to fit the f_water_model first.
    # For this dummy example, we'll fit it with some random data.
    # In the Optuna objective, f_water_model will be fitted on the actual data.

    # Dummy data for f_water model training
    # X should match the features prepared in _prepare_f_water_features and run method
    # [Observed_Difference_t, current_nutritional_vector, historical_nutritional_features, historical_pds_features]

    # Let's calculate the expected number of features for f_water_model
    nutritional_cols_count = len(
        [col for col in dummy_df.columns if col not in ["Pds", "sport"]]
    )
    expected_f_water_features_count = (
        1
        + nutritional_cols_count
        + (nutritional_cols_count * model.look_back_window)
        + model.look_back_window
    )

    X_f_water_train_dummy = np.random.rand(
        len(dummy_df) - model.look_back_window, expected_f_water_features_count
    )
    y_f_water_train_dummy = np.random.rand(
        len(dummy_df) - model.look_back_window
    )  # Dummy target 'p'

    model.f_water_model.fit(X_f_water_train_dummy, y_f_water_train_dummy)

    results = model.run(dummy_df)

    print("\nModel Run Results (first 5 entries):")
    for key, series in results.items():
        print(f"\n--- {key} ---")
        print(series.head())

    # Sanity check: sum of W_act and WR should be close to W_obs
    print("\nSanity Check: W_act + WR vs W_obs")
    print((results["W_act"] + results["WR"] - results["W_obs"]).head())
