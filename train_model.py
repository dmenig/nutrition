import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import json

from build_features import build_features

K_cal_kg = 7700


class FinalModel(nn.Module):
    def __init__(
        self, nutrition_input_size, initial_weight_guess, hidden_size=128, num_layers=2
    ):
        super(FinalModel, self).__init__()

        self.gru = nn.GRU(
            nutrition_input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )

        self.initial_metabolism = nn.Parameter(torch.tensor(2.5))
        self.initial_adj_weight = nn.Parameter(torch.tensor(initial_weight_guess))

        head_input_size = hidden_size + nutrition_input_size

        # Head to predict the daily *increment* for metabolism
        self.metabolism_increment_head = nn.Sequential(
            nn.Linear(head_input_size, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, nutrition_data):
        batch_size, seq_len, _ = nutrition_data.size()
        gru_out, _ = self.gru(nutrition_data)

        base_metabolisms = []

        current_metabolism = self.initial_metabolism.expand(batch_size, 1)

        for t in range(seq_len):
            current_gru_out = gru_out[:, t, :]
            current_nutrition = nutrition_data[:, t, :]
            combined_input = torch.cat((current_gru_out, current_nutrition), dim=-1)

            # Predict metabolism increment for smoothness
            metabolism_increment = (
                torch.tanh(self.metabolism_increment_head(combined_input)) * 0.125
            )  # Max 125 kcal change/day
            current_metabolism = current_metabolism + metabolism_increment
            base_metabolisms.append(current_metabolism)

        return torch.stack(base_metabolisms, dim=1)


def reconstruct_trajectory(
    observed_weights,
    base_metabolisms,
    nutrition_data,
    sport_data,
    initial_adj_weight,
    normalization_stats,
):
    batch_size, seq_len = observed_weights.shape

    # De-normalize data for physics calculations
    calories_stats = normalization_stats["calories"]
    calories_in_unnormalized = (
        nutrition_data[:, :, 0] * calories_stats["std"] + calories_stats["mean"]
    )

    sport_stats = normalization_stats["sport"]
    sport_data_unnormalized = (
        sport_data.squeeze(-1) * sport_stats["std"] + sport_stats["mean"]
    )

    calories_delta = (
        calories_in_unnormalized
        - sport_data_unnormalized
        - base_metabolisms.squeeze(-1) * 1000
    )

    # Calculate the predicted weight trajectory from the model's outputs
    w_adj_pred_list = [initial_adj_weight.expand(batch_size)]
    for t in range(1, seq_len):
        weight_change = calories_delta[:, t - 1] / K_cal_kg
        w_adj_t = w_adj_pred_list[t - 1] + weight_change
        w_adj_pred_list.append(w_adj_t)

    w_adj_pred = torch.stack(w_adj_pred_list, dim=1)
    w_pred = w_adj_pred

    return w_pred, w_adj_pred


def calculate_loss(
    observed_weights,
    predicted_observed_weight,
):
    # Loss component 1: Match the observed weight, with a gentle weight on recent data
    seq_len = observed_weights.shape[1]
    time_weights = torch.linspace(
        0.8, 1.2, steps=seq_len, device=observed_weights.device
    ).unsqueeze(0)
    squared_errors = (predicted_observed_weight - observed_weights) ** 2
    loss_fit = torch.mean(squared_errors * time_weights)

    total_loss = loss_fit

    return total_loss, {"loss_fit": loss_fit.item()}


def train_and_save_model(
    food_data_df,
    sport_data_df,
    model_save_path="models/recurrent_model.pth",
    params_save_path="models/best_params.json",
):
    # Combine food and sport data into a single DataFrame for feature building
    # Assuming food_data_df and sport_data_df have a common 'date' or 'timestamp' column
    # and can be merged or processed by build_features.py
    # For now, we'll pass them separately and adjust build_features if needed.

    # This part needs to be adapted based on how build_features expects its input
    # and how food_data_df and sport_data_df are structured.
    # For simplicity, let's assume build_features can take these two dataframes
    # and produce the features_df. If not, we'll need to preprocess them here.

    # For now, let's assume build_features can take raw dataframes and process them.
    # If build_features.py's main function is designed to read from files,
    # we'll need to refactor it or create a new function in build_features.py
    # that accepts dataframes directly.

    # Given the current train_model.py, it calls build_features.main() which
    # likely reads from some default location or generates data.
    # We need to ensure build_features can accept our fetched data.

    # For now, let's assume build_features_main can be adapted or we'll create a new function.
    # Let's assume for now that build_features.py has a function `build_features_from_dfs`
    # that takes food_data_df and sport_data_df and returns features_df.

    # If build_features.py's main function is designed to read from files,
    # we'll need to refactor it or create a new function in build_features.py
    # that accepts dataframes directly.

    # For now, let's call build_features.main() and assume it can handle the data.
    # This is a temporary placeholder and will need refinement.
    print("Calling build_features.main() to get features_df...")
    from build_features import main as build_features_main

    features_df = build_features_main(
        journal_path="data/processed_journal.csv",
        variables_path="data/variables.csv",
    )
    print(f"Features DataFrame loaded. Shape: {features_df.shape}")

    print("Starting data cleaning and normalization...")
    # Data cleaning and normalization
    nutrition_cols = ["calories", "carbs", "sugar", "sel", "alcool", "water"]
    for col in nutrition_cols + ["pds", "sport"]:
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(0)
    print("Data cleaning and normalization complete.")

    normalization_stats = {}
    for col in nutrition_cols + ["sport"]:
        mean, std = features_df[col].mean(), features_df[col].std()
        if std == 0:
            std = 1
        features_df[col] = (features_df[col] - mean) / std
        normalization_stats[col] = {"mean": mean, "std": std}

    weight_mean = features_df["pds"].mean()
    normalization_stats["pds"] = {"mean": weight_mean}
    features_df["pds_normalized"] = features_df["pds"] - weight_mean

    # Tensors
    observed_weights = torch.tensor(
        features_df["pds_normalized"].values, dtype=torch.float32
    ).unsqueeze(0)
    nutrition_data = torch.tensor(
        features_df[nutrition_cols].values, dtype=torch.float32
    ).unsqueeze(0)
    sport_data = torch.tensor(
        features_df["sport"].values, dtype=torch.float32
    ).unsqueeze(0)

    # Model, Optimizer
    initial_weight_guess = observed_weights[:, :5].mean().item()
    model = FinalModel(nutrition_data.shape[2], initial_weight_guess)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=50
    )

    print("Starting training with Final, Stable Model...")
    for epoch in range(600):
        optimizer.zero_grad()

        base_metabolisms = model(nutrition_data)

        predicted_observed_weight, _ = reconstruct_trajectory(
            observed_weights,
            base_metabolisms,
            nutrition_data,
            sport_data,
            model.initial_adj_weight,
            normalization_stats,
        )

        loss, loss_components = calculate_loss(
            observed_weights, predicted_observed_weight
        )

        if torch.isnan(loss):
            print(f"NaN detected at epoch {epoch}, stopping training.")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            print(f"  Components: {loss_components}")

    print("Training complete.")

    # --- Save Results ---
    print(f"Checking if model directory exists: {os.path.dirname(model_save_path)}")
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
        print(f"Created model directory: {os.path.dirname(model_save_path)}")
    else:
        print(f"Model directory already exists: {os.path.dirname(model_save_path)}")

    print(f"Checking if params directory exists: {os.path.dirname(params_save_path)}")
    if not os.path.exists(os.path.dirname(params_save_path)):
        os.makedirs(os.path.dirname(params_save_path))
        print(f"Created params directory: {os.path.dirname(params_save_path)}")
    else:
        print(f"Params directory already exists: {os.path.dirname(params_save_path)}")

    with torch.no_grad():
        final_metabolisms = model(nutrition_data)
        _, final_w_adj = reconstruct_trajectory(
            observed_weights,
            final_metabolisms,
            nutrition_data,
            sport_data,
            model.initial_adj_weight,
            normalization_stats,
        )

    results_df = features_df.copy()
    results_df["M_base"] = final_metabolisms.squeeze().numpy() * 1000
    results_df["W_obs"] = features_df["pds"]
    results_df["W_adj_pred"] = final_w_adj.squeeze().numpy() + weight_mean

    results_df.to_csv(
        "data/final_results.csv", index=False
    )  # This can remain as is, or be made configurable
    print(f"Attempting to save model to {model_save_path}...")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model state dictionary saved to {model_save_path}.")

    # Also export a lightweight NumPy weights file for torch-free inference
    try:
        import numpy as np
        npz_path = os.path.join(os.path.dirname(model_save_path), "recurrent_model_np.npz")
        # Collect state dict tensors into numpy arrays with stable names
        sd = model.state_dict()
        np.savez(
            npz_path,
            **{
                "gru.weight_ih_l0": sd.get("gru.weight_ih_l0", torch.empty(0)).cpu().numpy(),
                "gru.weight_hh_l0": sd.get("gru.weight_hh_l0", torch.empty(0)).cpu().numpy(),
                "gru.bias_ih_l0": sd.get("gru.bias_ih_l0", torch.empty(0)).cpu().numpy(),
                "gru.bias_hh_l0": sd.get("gru.bias_hh_l0", torch.empty(0)).cpu().numpy(),
                # Optional second layer
                "gru.weight_ih_l1": sd.get("gru.weight_ih_l1", torch.empty(0)).cpu().numpy(),
                "gru.weight_hh_l1": sd.get("gru.weight_hh_l1", torch.empty(0)).cpu().numpy(),
                "gru.bias_ih_l1": sd.get("gru.bias_ih_l1", torch.empty(0)).cpu().numpy(),
                "gru.bias_hh_l1": sd.get("gru.bias_hh_l1", torch.empty(0)).cpu().numpy(),
                # Head
                "head.0.weight": sd.get("metabolism_increment_head.0.weight").cpu().numpy(),
                "head.0.bias": sd.get("metabolism_increment_head.0.bias").cpu().numpy(),
                "head.2.weight": sd.get("metabolism_increment_head.2.weight").cpu().numpy(),
                "head.2.bias": sd.get("metabolism_increment_head.2.bias").cpu().numpy(),
                # Scalars
                "initial_metabolism": sd.get("initial_metabolism").cpu().numpy(),
                "initial_adj_weight": sd.get("initial_adj_weight").cpu().numpy(),
            }
        )
        print(f"Exported numpy weights to {npz_path}")
    except Exception as _e:
        print(f"Warning: failed to export numpy weights: {_e}")

    print(f"Attempting to save parameters to {params_save_path}...")
    with open(params_save_path, "w") as f:
        json.dump(
            {"architecture": "final_model", "normalization": normalization_stats},
            f,
            indent=4,
        )
    print(f"Parameters saved to {params_save_path}.")

    print(f"Model saved to {model_save_path} and parameters to {params_save_path}.")

    return model, {"architecture": "final_model", "normalization": normalization_stats}


if __name__ == "__main__":
    # When run as main, it will still use build_features.main() without arguments
    # which implies it will generate or read data as it currently does.
    # This ensures backward compatibility for direct execution of train_model.py
    from build_features import main as build_features_main

    features_df = build_features_main(
        journal_path="data/processed_journal.csv",
        variables_path="data/variables.csv",
    )

    # Data cleaning and normalization (repeated from train_and_save_model for standalone execution)
    nutrition_cols = ["calories", "carbs", "sugar", "sel", "alcool", "water"]
    for col in nutrition_cols + ["pds", "sport"]:
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(0)

    normalization_stats = {}
    for col in nutrition_cols + ["sport"]:
        mean, std = features_df[col].mean(), features_df[col].std()
        if std == 0:
            std = 1
        features_df[col] = (features_df[col] - mean) / std
        normalization_stats[col] = {"mean": mean, "std": std}

    weight_mean = features_df["pds"].mean()
    normalization_stats["pds"] = {"mean": weight_mean}
    features_df["pds_normalized"] = features_df["pds"] - weight_mean

    # Tensors
    observed_weights = torch.tensor(
        features_df["pds_normalized"].values, dtype=torch.float32
    ).unsqueeze(0)
    nutrition_data = torch.tensor(
        features_df[nutrition_cols].values, dtype=torch.float32
    ).unsqueeze(0)
    sport_data = torch.tensor(
        features_df["sport"].values, dtype=torch.float32
    ).unsqueeze(0)

    # Model, Optimizer
    initial_weight_guess = observed_weights[:, :5].mean().item()
    model = FinalModel(nutrition_data.shape[2], initial_weight_guess)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=50
    )

    print("Starting training with Final, Stable Model (standalone execution)...")
    for epoch in range(600):
        optimizer.zero_grad()

        base_metabolisms = model(nutrition_data)

        predicted_observed_weight, _ = reconstruct_trajectory(
            observed_weights,
            base_metabolisms,
            nutrition_data,
            sport_data,
            model.initial_adj_weight,
            normalization_stats,
        )

        loss, loss_components = calculate_loss(
            observed_weights, predicted_observed_weight
        )

        if torch.isnan(loss):
            print(f"NaN detected at epoch {epoch}, stopping training.")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            print(f"  Components: {loss_components}")

    print("Training complete (standalone execution).")

    # --- Save Results (standalone execution) ---
    # Ensure the 'models' directory exists
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")

    with torch.no_grad():
        final_metabolisms = model(nutrition_data)
        _, final_w_adj = reconstruct_trajectory(
            observed_weights,
            final_metabolisms,
            nutrition_data,
            sport_data,
            model.initial_adj_weight,
            normalization_stats,
        )

    results_df = features_df.copy()
    results_df["M_base"] = final_metabolisms.squeeze().numpy() * 1000
    results_df["W_obs"] = features_df["pds"]
    results_df["W_adj_pred"] = final_w_adj.squeeze().numpy() + weight_mean

    results_df.to_csv("data/final_results.csv", index=False)
    torch.save(model.state_dict(), os.path.join(models_dir, "recurrent_model.pth"))
    with open(os.path.join(models_dir, "best_params.json"), "w") as f:
        json.dump(
            {"architecture": "final_model", "normalization": normalization_stats},
            f,
            indent=4,
        )

    print("Final results and stable model saved (standalone execution).")
