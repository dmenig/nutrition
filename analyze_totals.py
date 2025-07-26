import pandas as pd

# Load the results
df = pd.read_csv('data/final_results.csv')

print("=== MODEL RESULTS ANALYSIS ===")
print(f"Total number of days: {len(df)}")
print()

# Calculate total calories delta (calories in - calories out)
total_calories_delta = (df['C_in_t'] - df['C_exp_t']).sum()
print(f"Total calories delta: {total_calories_delta:.1f} kcal")

# Calculate actual weight change
initial_weight = df['W_obs'].iloc[0]
final_weight = df['W_obs'].iloc[-1] 
actual_weight_change = final_weight - initial_weight
print(f"Actual weight change: {actual_weight_change:.3f} kg")
print(f"  Initial weight: {initial_weight:.3f} kg")
print(f"  Final weight: {final_weight:.3f} kg")

# Calculate expected weight change from physics (7700 kcal = 1 kg)
expected_weight_change_from_calories = total_calories_delta / 7700
print(f"Expected weight change from calories: {expected_weight_change_from_calories:.3f} kg")

# Calculate the difference
physics_match_error = abs(actual_weight_change - expected_weight_change_from_calories)
print(f"Physics match error: {physics_match_error:.3f} kg")
print(f"Physics match error in kcal: {physics_match_error * 7700:.1f} kcal")

# Calculate relative error
relative_error_pct = (physics_match_error / abs(actual_weight_change)) * 100 if actual_weight_change != 0 else 0
print(f"Relative error: {relative_error_pct:.1f}%")

print()
print("=== ADJUSTED WEIGHT ANALYSIS (removing water retention) ===")
# Check with adjusted weight (which should follow physics more closely)
initial_adj_weight = df['W_adj'].iloc[0]
final_adj_weight = df['W_adj'].iloc[-1]
adj_weight_change = final_adj_weight - initial_adj_weight
print(f"Adjusted weight change: {adj_weight_change:.3f} kg")
print(f"  Initial adjusted weight: {initial_adj_weight:.3f} kg") 
print(f"  Final adjusted weight: {final_adj_weight:.3f} kg")

# Physics match for adjusted weight
adj_physics_match_error = abs(adj_weight_change - expected_weight_change_from_calories)
print(f"Adjusted physics match error: {adj_physics_match_error:.3f} kg")
print(f"Adjusted physics match error in kcal: {adj_physics_match_error * 7700:.1f} kcal")

# Calculate relative error for adjusted weight
adj_relative_error_pct = (adj_physics_match_error / abs(adj_weight_change)) * 100 if adj_weight_change != 0 else 0
print(f"Adjusted relative error: {adj_relative_error_pct:.1f}%")

print()
print("=== WATER RETENTION ANALYSIS ===")
initial_water = df['WR'].iloc[0]
final_water = df['WR'].iloc[-1]
water_change = final_water - initial_water
print(f"Water retention change: {water_change:.3f} kg")
print(f"  Initial water retention: {initial_water:.3f} kg")
print(f"  Final water retention: {final_water:.3f} kg")

print()
print("=== SUMMARY ===")
if adj_physics_match_error < 0.5:  # Less than 500g error
    print("✅ PHYSICS CONSTRAINT WELL SATISFIED")
    print("The model successfully learned that weight change must equal calories delta / 7700")
elif adj_physics_match_error < 1.0:  # Less than 1kg error
    print("⚠️  PHYSICS CONSTRAINT MODERATELY SATISFIED")
    print("The model partially learned the physics constraint but has some error")
else:
    print("❌ PHYSICS CONSTRAINT NOT WELL SATISFIED")
    print("The model did not successfully learn the physics constraint")

print(f"Expected: {expected_weight_change_from_calories:.3f} kg change")
print(f"Actual (adjusted): {adj_weight_change:.3f} kg change")
print(f"Error: {adj_physics_match_error:.3f} kg ({adj_physics_match_error * 7700:.0f} kcal)") 