import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the results
try:
    df = pd.read_csv('data/final_results.csv')
except FileNotFoundError:
    print("Error: 'data/final_results.csv' not found.")
    print("Please run train_model.py first to generate the results file.")
    exit()

df['time_index'] = range(len(df))
plot_index = 'time_index'

# De-normalize calories and sport for plotting
# Note: This assumes 'best_params.json' was saved by the training script
import json
with open('best_params.json', 'r') as f:
    params = json.load(f)
    normalization_stats = params['normalization']

df['calories_unnorm'] = df['calories'] * normalization_stats['calories']['std'] + normalization_stats['calories']['mean']
df['sport_unnorm'] = df['sport'] * normalization_stats['sport']['std'] + normalization_stats['sport']['mean']
df['C_exp_t'] = df['M_base'] + df['sport_unnorm']


# --- Create Plots ---
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle('Final Model Performance Analysis', fontsize=16)

# Plot 1: Observed vs. Predicted Adjusted Weight
df['W_pred_final'] = df['W_adj_pred'] + df['WR']
axes[0, 0].plot(df[plot_index], df['W_obs'], 'k-', linewidth=2.5, alpha=0.8, label='Observed Weight (Ground Truth)')
axes[0, 0].plot(df[plot_index], df['W_pred_final'], 'r:', linewidth=2, label='Final Predicted Weight (Adj + WR)')
axes[0, 0].plot(df[plot_index], df['W_adj_pred'], 'b--', linewidth=1.5, alpha=0.7, label='Predicted Adjusted Weight (Trend)')
axes[0, 0].set_title('Weight Prediction vs. Ground Truth')
axes[0, 0].set_ylabel('Weight (kg)')
axes[0, 0].set_xlabel('Days')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Predicted Water Retention
axes[0, 1].plot(df[plot_index], df['WR'], 'g-', linewidth=2, label='Predicted Water Retention')
axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.5)
axes[0, 1].set_title('Predicted Water Retention')
axes[0, 1].set_ylabel('Water Retention (kg)')
axes[0, 1].set_xlabel('Days')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Predicted Base Metabolism
axes[1, 0].plot(df[plot_index], df['M_base'], 'm-', linewidth=2, label='Predicted Base Metabolism')
axes[1, 0].set_title('Predicted Base Metabolism Over Time')
axes[1, 0].set_ylabel('Metabolism (kcal/day)')
axes[1, 0].set_xlabel('Days')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Daily Energy Balance
axes[1, 1].plot(df[plot_index], df['calories_unnorm'], 'b-', linewidth=2, label='Calories Ingested', alpha=0.8)
axes[1, 1].plot(df[plot_index], df['C_exp_t'], 'r-', linewidth=2, label='Calories Expensed (Metabolism + Sport)', alpha=0.8)
axes[1, 1].fill_between(df[plot_index], df['calories_unnorm'], df['C_exp_t'], 
                        where=(df['calories_unnorm'] >= df['C_exp_t']), color='red', alpha=0.2, label='Surplus')
axes[1, 1].fill_between(df[plot_index], df['calories_unnorm'], df['C_exp_t'], 
                        where=(df['calories_unnorm'] < df['C_exp_t']), color='green', alpha=0.2, label='Deficit')
axes[1, 1].set_title('Daily Energy Balance')
axes[1, 1].set_ylabel('Calories (kcal/day)')
axes[1, 1].set_xlabel('Days')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('model_analysis_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Print Summary Statistics ---
# Calculate the "true" WR needed to make physics work with the predicted metabolism
df['C_delta'] = df['calories_unnorm'] - df['C_exp_t']
w_adj_physics = [df['W_obs'].iloc[0]]
for t in range(1, len(df)):
    w_change = df['C_delta'].iloc[t-1] / 7700
    w_adj_physics.append(w_adj_physics[t-1] + w_change)

df['WR_physics'] = df['W_obs'] - w_adj_physics
wr_rmse = np.sqrt(np.mean((df['WR_physics'] - df['WR'])**2))


rmse = np.sqrt(np.mean((df['W_obs'] - (df['W_adj_pred'] + df['WR']))**2))

print("="*40)
print("      Final Model Performance Summary")
print("="*40)
print(f"Overall Weight Prediction RMSE: {rmse:.4f} kg")
print(f"Water Retention Prediction RMSE: {wr_rmse:.4f} kg")
print("-"*40)

print("\n=== Model Internals Summary ===")
print(f"Predicted Metabolism Range: {df['M_base'].min():.1f} - {df['M_base'].max():.1f} kcal/day")
print(f"Predicted Metabolism Mean:  {df['M_base'].mean():.1f} kcal/day")
print(f"Predicted WR Range:         {df['WR'].min():.2f} - {df['WR'].max():.2f} kg")
print(f"Predicted WR Mean:          {df['WR'].mean():.2f} kg")

correlation = df[['W_obs', 'W_adj_pred']].corr().iloc[0, 1]
print(f"\nCorrelation between Observed and Adj. Weight: {correlation:.4f}")
print("="*40) 