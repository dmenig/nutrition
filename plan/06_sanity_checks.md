# Part 6: Sanity Checks and Remediation

**Goal:** Define a quantitative scoring system for the model's sanity checks and a clear plan for addressing failures. This ensures that the selected model is not just optimized for a mathematical loss function, but is also physically and physiologically plausible.

---

## 1. Quantitative Scoring System

The following metrics will be calculated on the output of the `analyze_results.py` script. These scores will be used to evaluate models during the loss weight grid search and to provide a final assessment of the chosen model.

### 1.1. Metabolism Range Score

*   **Concept:** The model's estimated base metabolism (`M_base`) should remain within a physiologically reasonable range for a healthy adult.
*   **Range:** 1400 - 3200 kcal/day.
*   **Metric:** Percentage of days where `M_base` is within the defined range.
*   **Calculation:** `Score_meta = 100 * (Number of days where 1400 <= M_base <= 3200) / (Total number of days)`
*   **Success Criterion:** `Score_meta >= 98%`. A near-perfect score is required, as deviations suggest a fundamental model instability.

### 1.2. Weight Smoothing Score

*   **Concept:** The "actual" weight (`W_act`) should be a less volatile, smoothed version of the noisy observed weight (`pds`).
*   **Metric:** Ratio of the standard deviation of the daily changes in `W_act` to the standard deviation of the daily changes in `pds`.
*   **Calculation:** `Score_smooth = std(diff(W_act)) / std(diff(pds))`
*   **Success Criterion:** `Score_smooth < 0.5`. The variation in the actual weight should be less than half the variation of the observed weight, indicating significant smoothing.

### 1.3. Calorie Delta Correlation Score

*   **Concept:** Over the long term, the change in actual weight should be strongly correlated with the cumulative energy balance (calorie delta). This is the first principle of thermodynamics applied to weight change.
*   **Metric:** Pearson correlation coefficient between the 30-day rolling cumulative calorie delta and the 30-day change in `W_act`.
*   **Calculation:**
    1.  `Calorie_Delta_t = C_in_t - C_exp_t`
    2.  `Cumulative_Delta_30d_t = Calorie_Delta.rolling(window=30).sum()`
    3.  `W_act_Change_30d_t = W_act.diff(30)`
    4.  `Score_corr = pearson_correlation(Cumulative_Delta_30d_t, W_act_Change_30d_t)`
*   **Success Criterion:** `Score_corr > 0.75`. A strong positive correlation is expected, confirming the model adheres to basic energy balance principles.

---

## 2. Remediation Plan

If a model fails one or more of the sanity checks, the following steps will be taken. The goal is not just to pass the check, but to understand *why* it failed.

### 2.1. Failure of Metabolism Range (`Score_meta < 98%`)

*   **Primary Cause:** The `L_meta` loss term is not strong enough to prevent unrealistic fluctuations in `M_base`. The model is sacrificing metabolic stability to minimize the water retention term.
*   **Remediation Steps:**
    1.  **Increase `w_meta`:** The primary solution. Increase the weight of the metabolism smoothness penalty (`w_meta`) in `loss_weights.json`. This will force the optimizer to prioritize a stable `M_base`.
    2.  **Adjust `alpha`:** Decrease the `alpha` parameter in `best_params.json`. A smaller `alpha` gives more weight to the historical metabolism (`M_t-1`), making it change more slowly.
    3.  **Data Investigation:** Check for extreme outliers in the input `features.csv` (e.g., a day with 10,000 calories consumed or burned) that could be forcing the metabolism to compensate drastically.

### 2.2. Failure of Weight Smoothing (`Score_smooth >= 0.5`)

*   **Primary Cause:** This is almost always a symptom of a fluctuating metabolism. If `M_base` is unstable, `C_exp` will be unstable, causing `W_act` to be unstable.
*   **Remediation Steps:**
    1.  **Follow Metabolism Remediation:** Address this by applying the same remediation steps as for the `Metabolism Range` failure. A smoother `M_base` will directly lead to a smoother `W_act`.
    2.  **Check `K_cal_kg`:** While less likely, an incorrect `K_cal_kg` value could amplify the effect of calorie deltas on weight, increasing volatility. Re-evaluate this constant if other steps fail.

### 2.3. Failure of Calorie Delta Correlation (`Score_corr <= 0.75`)

*   **Primary Cause:** The model has lost the fundamental connection between energy balance and weight change. The `WR` (water retention) term is likely dominating the prediction, or the metabolism is fluctuating in a way that decouples it from the calorie delta.
*   **Remediation Steps:**
    1.  **Decrease `w_water`:** Decrease the weight of the water retention penalty (`w_water`) in `loss_weights.json`. This gives the model more freedom to use the `WR` term to explain short-term noise, allowing `W_act` to follow the long-term energy trend more closely.
    2.  **Increase `w_meta`:** A more stable metabolism (higher `w_meta`) can reinforce the long-term trend, improving the correlation.
    3.  **Lengthen `look_back_window`:** A longer `look_back_window` for the water retention model might help it better distinguish between short-term noise and long-term trends, preventing it from interfering with the core energy balance.
    4.  **Data Integrity Check:** This failure could point to a severe issue in the `features.csv` data. Manually inspect the `calories` and `sport` columns for systematic errors or biases.

---

## 3. Workflow Integration

This plan will be implemented within the `train_model.py` script.

```mermaid
graph TD
    A[Start Grid Search for Loss Weights] --> B{For each w_meta, w_water pair};
    B --> C[Run Optuna Study];
    C --> D[Get Best Model Params];
    D --> E[Run Simulation with Best Params];
    E --> F{Calculate Sanity Scores};
    F --> G{All Scores Pass?};
    G -- Yes --> H[Store Score and Params];
    G -- No --> B;
    H --> I[Select Best Overall w_meta, w_water];
    I --> J[Run Final Training];
    J --> K[Save Artifacts: best_params.json, loss_weights.json];