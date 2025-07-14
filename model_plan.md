# Plan for Weight and Metabolism Prediction Model

This document outlines the plan to build a model that predicts actual body weight and water retention based on historical data and observed weight. The model will be tuned using Optuna to meet several objectives simultaneously.

## 1. Core Objective

The primary goal is to decompose the daily observed weight (`W_obs`) into two components:
1.  **Actual Weight (`W_act`)**: The underlying "true" body weight, which should change based on energy balance.
2.  **Water Retention (`WR`)**: Transient fluctuations due to water, which should sum to zero over long periods.

The core relationship is: `W_obs(t) = W_act(t) + WR(t)`.

## 2. Model Architecture

The model will be structured as an iterative process that updates its state at each time step `t`.

### Model State
- **`W_act(t)`**: The estimated actual weight for day `t`.
- **`M_base(t)`**: The estimated base metabolic rate (in kcal/day) for day `t`. This is the "almost constant" component that adapts slowly over time.

### Inputs for day `t`
- `W_obs(t)`: Observed weight from the scale.
- `C_exp_activity(t)`: Calories expended through planned activity.
- `Nutritional_Vector(t)`: A vector of all calculated nutritional values for the day (e.g., `[C_in, Protein, Carbs, Fat, Alcohol, ...]`).
- `Historic Data`: A window of past data for all inputs.

### Process Flow

A diagram of the daily update logic:

```mermaid
graph TD
    A[State at t-1: W_act(t-1), M_base(t-1)] --> B{1. Predict Weight from Calories};
    B --> C{2. Calculate Observed Difference};
    subgraph Inputs for Day t
        D[W_obs(t), Nutritional_Vector(t), C_exp_activity(t)]
    end
    D --> B;
    D --> C;
    subgraph Historic Data Window
        F_hist[Past Nutritional Vectors]
        W_hist[Past Observed Weights]
    end
    C --> E{3. Attribute Difference};
    F_hist --> E;
    W_hist --> E;
    E -- Proportion 'p' --> F[WR(t) = p * Diff];
    E -- Proportion '1-p' --> G[W_act_correction(t) = (1-p) * Diff];
    B -- W_act_predicted --> H{4. Finalize W_act(t)};
    G --> H;
    H --> I{5. Update Metabolism M_base(t)};
    F --> J[Output: WR(t)];
    H --> K[Output: W_act(t)];
    I --> L[New State at t: W_act(t), M_base(t)];
```

### Step-by-Step Calculation

1.  **Predict Weight Change from Calories**:
    - `C_in(t)` is extracted from `Nutritional_Vector(t)`.
    - `C_total_exp(t) = M_base(t-1) + C_exp_activity(t)`
    - `C_delta(t) = C_in(t) - C_total_exp(t)`
    - `W_act_predicted_from_calories(t) = W_act(t-1) + C_delta(t) / K_cal_kg`
    - `K_cal_kg` is a parameter to be tuned (approx. 7700 kcal/kg).

2.  **Calculate Observed Difference**:
    - This is the portion of weight change not explained by the initial calorie model.
    - `Observed_Difference(t) = W_obs(t) - W_act_predicted_from_calories(t)`

3.  **Attribute Difference to Water vs. Metabolism**:
    - A machine learning model `f_water` will determine the proportion `p(t)` of the difference to attribute to water. This is the "second component" from the prompt.
    - `p(t) = f_water(Observed_Difference(t), window(Nutritional_Vector), window(W_obs))`
    - The hyperparameters of this model (e.g., learning rate, tree depth) will be tuned by Optuna.

4.  **Calculate Final Weight and Water**:
    - `WR(t) = p(t) * Observed_Difference(t)`
    - `W_act_correction(t) = (1 - p(t)) * Observed_Difference(t)`
    - `W_act(t) = W_act_predicted_from_calories(t) + W_act_correction(t)`

5.  **Update Base Metabolism with EMA Smoothing**:
    - To prevent a volatile metabolism, the update is smoothed using an Exponential Moving Average (EMA).
    - First, calculate a "target" metabolism for the day:
      `M_base_target(t) = M_base(t-1) - (W_act_correction(t) * K_cal_kg)`
    - Then, update the metabolism by moving it a small fraction (`alpha`) towards the target:
      `M_base(t) = (1 - alpha) * M_base(t-1) + alpha * M_base_target(t)`
    - The `alpha` parameter will be tuned by Optuna to find the optimal smoothing factor.

## 3. Feature Engineering

Before running the model, a comprehensive feature set will be created.

1.  **Load Data**: Load `processed_journal.csv` and `variables.csv`.
2.  **Calculate Daily Nutrients**: For each day in the journal, iterate through all available nutrients in `variables.csv` (Calories, Protéines, Glucides, Lipides, Alcool, etc.) and use the `calculate_nutrient_from_formula` logic to compute the total daily intake for each.
3.  **Create Feature Matrix**: Assemble this data into a time-indexed DataFrame where each row is a day and columns represent the full `Nutritional_Vector(t)`.

## 4. Tuning with Optuna

Optuna will be used to find the optimal model parameters by minimizing a combined objective function.

### Parameters to Tune
- `K_cal_kg`: The energy equivalent of 1kg of body weight.
- **Hyperparameters of the `f_water` model**: This could include learning rate, number of estimators, layer sizes, etc., depending on the chosen model (e.g., Gradient Boosting, small NN).
- Initial `M_base(0)`.
- `alpha`: The smoothing factor for the EMA metabolism update.
- Weights for the loss components (`w_meta`, `w_accuracy`, `w_water`).
- The size of the look-back window for historical data.

### Objective Functions (Losses)

1.  **`L_meta` (Metabolism Stability)**: Penalizes rapid changes in metabolism.
    - `L_meta = sum( (M_base(t) - M_base(t-1))^2 )`

2.  **`L_accuracy` (Prediction Accuracy)**: Ensures the `W_act(t)` follows the underlying trend of the observed weight.
    - `L_accuracy = sum( (W_act(t) - smoothed(W_obs(t)))^2 )`
    - `smoothed(W_obs(t))` could be a moving average of the observed weight.

3.  **`L_water` (Water Retention Minimization)**: A regularization term to prevent the model from attributing all fluctuations to water.
    - `L_water = sum( WR(t)^2 )`

### Combined Objective
The function to minimize is a weighted sum of these losses:
`Total_Loss = w_meta * L_meta + w_accuracy * L_accuracy + w_water * L_water`

## 5. Sanity Check (Post-Tuning)

After the best model is found, a final test will be performed to ensure it respects the principle of energy conservation over a long period.

- **Test**: Verify that `W_act(T) - W_act(0)` is close to `sum(C_delta(t) for t=0..T) / K_cal_kg`.
- **Purpose**: This confirms that the model doesn't "invent" or "lose" weight over time and that long-term changes are correctly attributed to the cumulative calorie surplus or deficit.

## 6. Implementation Todo List