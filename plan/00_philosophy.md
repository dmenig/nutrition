# Part 0: Project Philosophy and Core Concepts

This document outlines the high-level goals, core principles, and overall philosophy of the weight prediction model. It serves as the conceptual foundation for the detailed implementation plans.

## 1. High-Level Goal

The primary objective is to create a model that decomposes a user's daily observed weight (`W_obs`) into two unobserved components:

1.  **Actual Weight (`W_act`)**: The user's "true" physiological weight, which changes slowly and rationally based on the fundamental principles of calorie balance.
2.  **Water Retention (`WR`)**: Short-term, noisy fluctuations caused by diet, exercise, and other transient factors.

The core relationship is: `W_obs_t = W_act_t + WR_t`.

## 2. Core Philosophy: Emergent Accuracy

The model's accuracy is not achieved by directly minimizing the difference between predicted and observed weight. Instead, accuracy emerges from the tension between two opposing, physically-motivated objectives:

*   **Metabolism Smoothness (`L_meta`):** We penalize rapid, unrealistic changes in the user's base metabolism. This enforces the principle that metabolism is a slow-moving physiological process.
*   **Water Retention Minimization (`L_water`):** We penalize the magnitude of the water retention component. This encourages the model to be parsimonious, using the `WR` term only to explain the noise that the core energy balance model cannot.

By finding the right balance between these two forces, the model is driven to find a physically plausible explanation for the observed weight data, leading to an accurate and interpretable result.

## 3. Development and Error Handling Philosophy

*   **No-Fail Policy:** All scripts are designed to fail immediately and loudly upon any error. There are no `try-except` blocks to mask bugs. This ensures that any deviation from expected behavior is surfaced and can be investigated directly.
*   **Maximum Reusability:** Logic is encapsulated into well-defined functions that are reused across different stages of the project (training, analysis, etc.). This ensures consistency and reduces the chance of error.
*   **TDD Mindset:** While not formally writing tests first in this planning stage, the implementation will be guided by a Test-Driven Development mindset, with each component having a corresponding, detailed test plan.

## 4. Overall Workflow

The project follows a clear, sequential workflow, with each step producing a well-defined artifact that serves as the input for the next.

```mermaid
graph TD
    A[Raw Data: processed_journal.csv] --> B(data_processor.py);
    B --> C[features.csv];
    C --> D(train_model.py);
    D --> E{Grid Search for Loss Weights};
    E --> F[loss_weights.json];
    D --> G{Optuna Study for Model Params};
    G --> H[best_params.json];
    C --> I(analyze_results.py);
    F --> I;
    H --> I;
    I --> J[Plots and Sanity Checks];