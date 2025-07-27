# Milestone 2: Core Logic & Model Integration (Revised)

This document outlines the tasks for implementing the core business logic, including the daily model retraining process and the API endpoints for visualization.

### **Objective:**
To have a fully functional backend that can log detailed data, provide daily summaries, serve data for native plots, and automatically retrain its own prediction model daily.

---

### **Tasks:**

#### 2.1: Implement Granular Food Log Endpoints
-   **Action:** Implement the full CRUD lifecycle for food logs with detailed editing.
-   **Endpoints:**
    -   `POST /api/v1/logs`: Create a new food log with `quantity` and `unit`.
    -   `GET /api/v1/logs?date=YYYY-MM-DD`: Retrieve logs for a specific day.
    -   `PUT /api/v1/logs/{log_id}`: **(New)** Update an existing food log's details (e.g., change quantity).
    -   `DELETE /api/v1/logs/{log_id}`: Delete a log entry.

#### 2.2: Implement Daily Summary Endpoint
-   **Action:** Create an efficient endpoint to get daily totals.
-   **Endpoint:** `GET /api/v1/logs/summary?date=YYYY-MM-DD`
-   **Logic:**
    -   This endpoint will perform a SQL `SUM()` aggregation grouped by date on the `food_logs` table.
    -   It will return the total `calories`, `protein_g`, `carbs_g`, and `fat_g` for the given day. This is highly efficient and avoids pulling all log entries into memory.

#### 2.3: Implement Model Retraining Job
-   **Action:** Create a script (`app/jobs/retrain_model.py`) that can be run to retrain the model.
-   **Logic:**
    1.  Fetch all historical food and sport data from the PostgreSQL database.
    2.  Run the existing `train_model.py` logic.
    3.  Save the newly trained model file (e.g., `model.pkl`) and the `best_params.json` to a persistent location (e.g., overwriting the existing ones in the deployed container).
-   **Endpoint:** Create a secure `POST /api/v1/admin/retrain` endpoint that can trigger this job manually. This endpoint should be protected (e.g., require a special admin API key).

#### 2.4: Implement Visualization Data Endpoints
-   **Action:** Create endpoints to provide the raw data needed for the native Android plots.
-   **Logic:** These endpoints will query the database, run the necessary calculations (similar to `plot_results.py`), and return the data in a clean JSON format.
-   **Endpoints:**
    -   `GET /api/v1/plots/weight`: Returns a time series of `W_obs` and `W_adj_pred`.
    -   `GET /api/v1/plots/metabolism`: Returns a time series of `M_base`.
    -   `GET /api/v1/plots/energy-balance`: Returns time series for `calories_unnorm` and `C_exp_t`.

#### 2.5: Update Prediction Service
-   **Action:** Modify the prediction service to use the live model.
-   **Logic:**
    -   The `PredictionService` will now load the `model.pkl` file from disk when the server starts.
    -   The `GET /api/v1/predict/latest` endpoint will use this live model for inference.
    -   Implement a mechanism to reload the model into memory after the daily retraining job completes, without requiring a server restart.