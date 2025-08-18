import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the results
try:
    df = pd.read_csv("data/final_results.csv")
except FileNotFoundError:
    print("Error: 'data/final_results.csv' not found.")
    print("Please run train_model.py first to generate the results file.")
    exit()

df["time_index"] = range(len(df))
plot_index = "time_index"

# Determine if calories/sport are normalized (z-scores) or absolute kcal.
# Heuristic: if the 95th percentile is < 1000, treat as normalized and de-normalize
# using models/best_params.json; otherwise pass-through.
import json

cal_raw = pd.to_numeric(df["calories"], errors="coerce")
sport_raw = pd.to_numeric(df["sport"], errors="coerce")

cal_p95 = cal_raw.quantile(0.95)
sport_p95 = sport_raw.quantile(0.95)

norm_stats = None
try:
    with open("models/best_params.json", "r") as f:
        params = json.load(f)
        norm_stats = params.get("normalization", {})
except Exception:
    norm_stats = None

def maybe_denorm(series: pd.Series, key: str) -> pd.Series:
    if norm_stats is None or key not in norm_stats:
        return pd.to_numeric(series, errors="coerce")
    mean = float(norm_stats[key].get("mean", 0.0))
    std = float(norm_stats[key].get("std", 1.0) or 1.0)
    return pd.to_numeric(series, errors="coerce") * std + mean

if cal_p95 is not None and cal_p95 < 1000:
    df["calories_unnorm"] = maybe_denorm(df["calories"], "calories")
else:
    df["calories_unnorm"] = cal_raw

if sport_p95 is not None and sport_p95 < 1000:
    df["sport_unnorm"] = maybe_denorm(df["sport"], "sport")
else:
    df["sport_unnorm"] = sport_raw

df["C_exp_t"] = pd.to_numeric(df["M_base"], errors="coerce") + df["sport_unnorm"]


# --- Create Plots ---
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle("Final Model Performance Analysis", fontsize=16)

# Plot 1: Observed vs. Predicted Adjusted Weight
axes[0, 0].plot(
    df[plot_index],
    df["W_obs"],
    "k-",
    linewidth=2.5,
    alpha=0.8,
    label="Observed Weight (Ground Truth)",
)
axes[0, 0].plot(
    df[plot_index],
    df["W_adj_pred"],
    "b--",
    linewidth=1.5,
    alpha=0.7,
    label="Predicted Adjusted Weight (Trend)",
)
axes[0, 0].set_title("Weight Prediction vs. Ground Truth")
axes[0, 0].set_ylabel("Weight (kg)")
axes[0, 0].set_xlabel("Days")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Predicted Base Metabolism
axes[0, 1].plot(
    df[plot_index], df["M_base"], "m-", linewidth=2, label="Predicted Base Metabolism"
)
axes[0, 1].set_title("Predicted Base Metabolism Over Time")
axes[0, 1].set_ylabel("Metabolism (kcal/day)")
axes[0, 1].set_xlabel("Days")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Daily Energy Balance
axes[1, 0].plot(
    df[plot_index],
    df["calories_unnorm"],
    "b-",
    linewidth=2,
    label="Calories Ingested",
    alpha=0.8,
)
axes[1, 0].plot(
    df[plot_index],
    df["C_exp_t"],
    "r-",
    linewidth=2,
    label="Calories Expensed (Metabolism + Sport)",
    alpha=0.8,
)
axes[1, 0].fill_between(
    df[plot_index],
    df["calories_unnorm"],
    df["C_exp_t"],
    where=(df["calories_unnorm"] >= df["C_exp_t"]),
    color="red",
    alpha=0.2,
    label="Surplus",
)
axes[1, 0].fill_between(
    df[plot_index],
    df["calories_unnorm"],
    df["C_exp_t"],
    where=(df["calories_unnorm"] < df["C_exp_t"]),
    color="green",
    alpha=0.2,
    label="Deficit",
)
axes[1, 0].set_title("Daily Energy Balance")
axes[1, 0].set_ylabel("Calories (kcal/day)")
axes[1, 0].set_xlabel("Days")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Remove the empty subplot
fig.delaxes(axes[1, 1])


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("model_analysis_plots.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Print Summary Statistics ---
# Calculate the "true" WR needed to make physics work with the predicted metabolism
df["C_delta"] = df["calories_unnorm"] - df["C_exp_t"]
w_adj_physics = [df["W_obs"].iloc[0]]
for t in range(1, len(df)):
    w_change = df["C_delta"].iloc[t - 1] / 7700
    w_adj_physics.append(w_adj_physics[t - 1] + w_change)

df["WR_physics"] = df["W_obs"] - w_adj_physics


rmse = np.sqrt(np.mean((df["W_obs"] - df["W_adj_pred"]) ** 2))

print("=" * 40)
print("      Final Model Performance Summary")
print("=" * 40)
print(f"Overall Weight Prediction RMSE: {rmse:.4f} kg")
print("-" * 40)

print("\n=== Model Internals Summary ===")
print(
    f"Predicted Metabolism Range: {df['M_base'].min():.1f} - {df['M_base'].max():.1f} kcal/day"
)
print(f"Predicted Metabolism Mean:  {df['M_base'].mean():.1f} kcal/day")

correlation = df[["W_obs", "W_adj_pred"]].corr().iloc[0, 1]
print(f"\nCorrelation between Observed and Adj. Weight: {correlation:.4f}")
print("=" * 40)
