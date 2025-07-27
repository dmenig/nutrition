# Milestone 4: Feature Enhancement (Revised)

This document outlines the tasks for enhancing the MVP with advanced, user-centric features.

### **Objective:**
To build a polished, highly usable app that streamlines the logging process and provides rich, interactive insights.

---

### **Tasks:**

#### 4.1: Implement Combined Food Search
-   **Action:** Create a unified search experience in the Android app.
-   **Logic:**
    -   When the user searches for a food, the app will query two sources in parallel:
        1.  The local `custom_foods` library via the Room DAO.
        2.  The Open Food Facts API.
    -   The UI will display results from both sources in a single, clearly labeled list (e.g., with a "My Foods" header).

#### 4.2: Implement Barcode Scanning
-   **Action:** Add barcode scanning for quick food entry.
-   **Technology:** Use **CameraX** and **ML Kit Vision**.
-   **Logic:**
    1.  On scan, query the Open Food Facts API.
    2.  If found, pre-fill the food logging form.
    3.  If not found, prompt the user to add it as a new custom food.

#### 4.3: Implement Smart Food Lists
-   **Action:** Add "Recent" and "Frequent" food lists to the logging screen.
-   **Logic:**
    -   **Recent Foods:** Query the local Room database for the last 20 unique food log entries.
    -   **Frequent Foods:** Use a SQL query to group by `food_name`, count occurrences, and return the top 10.
    -   Tapping an item will pre-fill the logging form.

#### 4.4: Enhance Interactive Plots
-   **Action:** Refine the native plotting screen.
-   **UI/UX:**
    -   Ensure pinch-to-zoom and panning are smooth.
    -   Add a "date range" selector to allow the user to view data over different time periods (e.g., last 7 days, last 30 days, all time).
    -   The `ViewModel` will pass the selected date range to the backend API to fetch the appropriate data slice.

#### 4.5: Implement Robust Offline Sync
-   **Action:** Ensure data integrity and synchronization.
-   **Technology:** Use Android's `WorkManager`.
-   **Logic:**
    1.  All new or edited logs (`FoodLog`, `SportActivity`, `CustomFood`) will be marked with a `synced = false` flag in the local Room database.
    2.  A `SyncWorker` will run periodically and on network change.
    3.  The worker will attempt to push all unsynced items to the backend.
    4.  Crucially, it will handle potential conflicts (e.g., an item was deleted on the server but edited locally) using a "last-write-wins" strategy based on a `last_modified` timestamp.