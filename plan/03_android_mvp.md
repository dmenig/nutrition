# Milestone 3: Android App MVP (Revised)

This document specifies the tasks for building a high-performance MVP, focusing on efficient data handling, a responsive UI, and core user flows for logging and editing.

### **Objective:**
To create a fast, functional Android app that allows detailed logging and editing of food and sport, with a highly responsive interface for viewing daily data.

---

### **Tasks:**

#### 3.1: Project & Data Layer Setup
-   **Action:** Set up the Android project with Hilt, Retrofit, and Room as previously defined.
-   **Entities:** Define Room entities for `FoodLog`, `SportActivity`, and `CustomFood`.
-   **DAOs:** Create DAOs with specific, performant queries.
    -   `getLogsForDay(date)`: Selects logs only for a specific day.
    -   `getDailySummary(date)`: Uses `SUM()` to calculate daily totals directly in the database.
-   **Repository:** Implement the repository pattern to be the single source of truth, abstracting the local database and network API calls.

#### 3.2: Implement a Performant Daily Log Screen
-   **Action:** Build the main screen for viewing and navigating daily data.
-   **UI:**
    -   A calendar view at the top to easily select any day.
    -   A summary section below the calendar that displays the output of the `getDailySummary` query for the selected date.
    -   A paginated list showing the individual food and sport logs for the selected day.
-   **Performance:** The app will only query and load data for the **selected day**, ensuring the UI is always fast and responsive, regardless of the total amount of data in the database.

#### 3.3: Implement Granular Logging and Editing
-   **Action:** Build the forms for adding and editing entries.
-   **UI:**
    -   **Food Entry Form:** Fields for `food_name`, `quantity`, and `unit`. A date/time picker will allow setting the precise `logged_at` timestamp.
    -   **Sport Entry Form:** Fields for `activity_name`, `duration`, and `calories_expended`.
-   **Logic:**
    -   When a user taps an existing log item, the app will navigate to the corresponding form, pre-filled with the data.
    -   Saving the form will either create a new entry or update an existing one.

#### 3.4: Implement Custom Food Management
-   **Action:** Build the UI for managing the user's personal food library.
-   **UI:**
    -   A screen showing a list of all custom foods.
    -   A form to add a new custom food with its nutritional information per 100g.
    -   The ability to delete a custom food.
-   **Integration:** When logging food, the user will be able to search both the Open Food Facts API and their personal `custom_foods` library.

#### 3.5: Implement Native Plotting Screen
-   **Action:** Create a dedicated screen for data visualization.
-   **Technology:** Use **MPAndroidChart** for rich, interactive, and performant native charts.
-   **UI:**
    -   A tabbed interface or a dropdown to switch between the different plots (Weight, Metabolism, Energy Balance).
    -   The chart view will support pinch-to-zoom and panning.
    -   Tapping on a data point will show a marker with the precise value and timestamp.
-   **ViewModel:** A `PlotsViewModel` will call the new backend endpoints (e.g., `/api/v1/plots/weight`) to get the raw data series and format it for the chart library.