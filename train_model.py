import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import json

from build_features import build_features

K_cal_kg = 7700


class FinalModel(nn.Module):
    def __init__(self, nutrition_input_size, initial_weight_guess, hidden_size=64, num_layers=2):
        super(FinalModel, self).__init__()
        
        self.gru = nn.GRU(
            nutrition_input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.initial_metabolism = nn.Parameter(torch.tensor(2.5))
        self.initial_adj_weight = nn.Parameter(torch.tensor(initial_weight_guess))
        
        head_input_size = hidden_size + nutrition_input_size
        
        # Head to predict the daily *increment* for metabolism
        self.metabolism_increment_head = nn.Sequential(
            nn.Linear(head_input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Head to predict water retention
        self.water_retention_head = nn.Sequential(
            nn.Linear(head_input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, nutrition_data):
        batch_size, seq_len, _ = nutrition_data.size()
        gru_out, _ = self.gru(nutrition_data)
        
        base_metabolisms = []
        water_retentions = []
        
        current_metabolism = self.initial_metabolism.expand(batch_size, 1)

        for t in range(seq_len):
            current_gru_out = gru_out[:, t, :]
            current_nutrition = nutrition_data[:, t, :]
            combined_input = torch.cat((current_gru_out, current_nutrition), dim=-1)
            
            # Predict metabolism increment for smoothness
            metabolism_increment = torch.tanh(self.metabolism_increment_head(combined_input)) * 0.075 # Max 75 kcal change/day
            current_metabolism = current_metabolism + metabolism_increment
            base_metabolisms.append(current_metabolism)
            
            # Predict water retention
            current_wr = torch.tanh(self.water_retention_head(combined_input)) * 2.5
            water_retentions.append(current_wr)
            
        return torch.stack(base_metabolisms, dim=1), torch.stack(water_retentions, dim=1)


def reconstruct_trajectory(
    observed_weights,
    base_metabolisms,
    predicted_water_retentions,
    nutrition_data,
    sport_data,
    initial_adj_weight,
    normalization_stats,
):
    batch_size, seq_len = observed_weights.shape
    
    # De-normalize data for physics calculations
    calories_stats = normalization_stats['calories']
    calories_in_unnormalized = nutrition_data[:, :, 0] * calories_stats['std'] + calories_stats['mean']
    
    sport_stats = normalization_stats['sport']
    sport_data_unnormalized = sport_data.squeeze(-1) * sport_stats['std'] + sport_stats['mean']
    
    calories_delta = calories_in_unnormalized - sport_data_unnormalized - base_metabolisms.squeeze(-1) * 1000

    # Calculate the predicted weight trajectory from the model's outputs
    w_adj_pred_list = [initial_adj_weight.expand(batch_size)]
    for t in range(1, seq_len):
        weight_change = calories_delta[:, t - 1] / K_cal_kg
        w_adj_t = w_adj_pred_list[t-1] + weight_change
        w_adj_pred_list.append(w_adj_t)
        
    w_adj_pred = torch.stack(w_adj_pred_list, dim=1)
    w_pred = w_adj_pred + predicted_water_retentions.squeeze(-1)
    
    return w_pred, w_adj_pred


def calculate_loss(
    observed_weights,
    predicted_observed_weight,
    predicted_water_retentions,
):
    # Loss component 1: Match the observed weight
    loss_fit = torch.mean((predicted_observed_weight - observed_weights) ** 2)
    
    # Loss component 2: Penalize non-zero mean water retention
    loss_wr_mean = torch.mean(predicted_water_retentions) ** 2
    
    # Combine losses with a weight for the mean penalty
    total_loss = loss_fit + 1.0 * loss_wr_mean
    
    return total_loss, {
        "loss_fit": loss_fit.item(),
        "loss_wr_mean": loss_wr_mean.item()
    }


def main():
    from build_features import main as build_features_main
    features_df = build_features_main()

    # Data cleaning and normalization
    nutrition_cols = ["calories", "carbs", "sugar", "sel", "alcool", "water"]
    for col in nutrition_cols + ["pds", "sport"]:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
        
    normalization_stats = {}
    for col in nutrition_cols + ["sport"]:
        mean, std = features_df[col].mean(), features_df[col].std()
        if std == 0: std = 1
        features_df[col] = (features_df[col] - mean) / std
        normalization_stats[col] = {"mean": mean, "std": std}
    
    weight_mean = features_df["pds"].mean()
    normalization_stats["pds"] = {"mean": weight_mean}
    features_df["pds_normalized"] = features_df["pds"] - weight_mean
    
    # Tensors
    observed_weights = torch.tensor(features_df["pds_normalized"].values, dtype=torch.float32).unsqueeze(0)
    nutrition_data = torch.tensor(features_df[nutrition_cols].values, dtype=torch.float32).unsqueeze(0)
    sport_data = torch.tensor(features_df["sport"].values, dtype=torch.float32).unsqueeze(0)

    # Model, Optimizer
    initial_weight_guess = observed_weights[:, :5].mean().item()
    model = FinalModel(nutrition_data.shape[2], initial_weight_guess)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Increased learning rate

    print("Starting training with Final, Stable Model...")
    for epoch in range(500):
        optimizer.zero_grad()

        base_metabolisms, water_retentions = model(nutrition_data)

        predicted_observed_weight, _ = reconstruct_trajectory(
            observed_weights, base_metabolisms, water_retentions, 
            nutrition_data, sport_data, model.initial_adj_weight, normalization_stats
        )
        
        loss, loss_components = calculate_loss(observed_weights, predicted_observed_weight, water_retentions)

        if torch.isnan(loss):
            print(f"NaN detected at epoch {epoch}, stopping training.")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            print(f"  Components: {loss_components}")

    print("Training complete.")

    # --- Save Results ---
    if not os.path.exists("data"): os.makedirs("data")

    with torch.no_grad():
        final_metabolisms, final_wr = model(nutrition_data)
        _, final_w_adj = reconstruct_trajectory(
            observed_weights, final_metabolisms, final_wr, 
            nutrition_data, sport_data, model.initial_adj_weight, normalization_stats
        )

    results_df = features_df.copy()
    results_df["M_base"] = final_metabolisms.squeeze().numpy() * 1000
    results_df["WR"] = final_wr.squeeze().numpy()
    results_df["W_obs"] = features_df["pds"]
    results_df["W_adj_pred"] = final_w_adj.squeeze().numpy() + weight_mean
    
    results_df.to_csv("data/final_results.csv", index=False)
    torch.save(model.state_dict(), "recurrent_model.pth")
    with open("best_params.json", "w") as f:
        json.dump({"architecture": "final_model", "normalization": normalization_stats}, f, indent=4)
        
    print("Final results and stable model saved.")


if __name__ == "__main__":
    main()
