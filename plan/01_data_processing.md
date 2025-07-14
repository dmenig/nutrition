# Part 1: Data Processing

**Goal:** Create a script that processes the raw data and produces a clean `features.csv` file.

**Files to Create/Modify:**
*   `build_features.py`
*   `tests/test_build_features.py`

**Implementation Details (`build_features.py`):**

This script will serve as an orchestrator, combining functionalities from existing, stable modules (`data_processor.py`, `nutrition_calculator.py`, `sport_formulas.py`) to produce the final `features.csv`. **It will not modify these modules.**

*   **Data Loading:** It will import and call the main data processing function from `data_processor.py` to get the initial DataFrame with parsed food nutrients.
*   **Sport Calorie Calculation:** It will import functions from `sport_formulas.py` and the `SafeSportFormulaEvaluator` from `utils.py` to parse the `Sport` column and calculate calorie expenditure. This logic will be contained within `build_features.py`.
*   **Column Naming:** The logic for cleaning column names (`strip_accents(nutrient.split(' / ')[0]).replace(" ", "_").lower()`) will be applied within this script after the initial data is loaded.
*   **Error Handling:** The script will adhere to the "no-fail" policy. Any error during the import or execution of functions from other modules will cause the script to fail immediately.
*   **Output:** The script will save the final, fully-processed DataFrame to `data/features.csv` with the following schema:
    *   **Index**: `Date` (type: `pd.DatetimeIndex`, sorted ascending).
    *   **Columns**:
        *   `pds` (float): The observed weight for the day.
        *   `sport` (float): Calories burned from exercise.
        *   `calories` (float): Total calculated calories consumed.
        *   `proteine` (float)
        *   `fat` (float)
        *   `sfat` (float)
        *   `carbs` (float)
        *   `sugar` (float)
        *   `free_sugar` (float)
        *   `fibres` (float)
        *   `sel` (float)
        *   `alcool` (float)
        *   `water` (float)

**Testing (`tests/test_build_features.py`):**

The test suite for `build_features.py` will be comprehensive, using a single, powerful test function that covers multiple edge cases by mocking the outputs of the imported functions.

*   **Test Setup:**
    *   A dummy `variables.csv` will be created programmatically within the test. It will contain a variety of food items with a full spectrum of nutrients.
        *   **Example Foods:** `farine`, `sucre`, `pain complet`, `fromage de chèvre`, `poulet`, `seven_up_mojito`.
        *   **Nutrients:** `Calories / 100g`, `Protéine`, `Fat`, `SFat`, `Carbs`, `Sugar`, `Free sugar`, `Fibres`, `Sel`, `Alcool`, `Water`.
    *   A dummy `processed_journal.csv` will also be created, containing several rows with realistic and complex data.
        *   **Row 1 (Complex Food & Sport):**
            *   `Nourriture`: "150 * (pain complet * 0.5 + fromage de chèvre * 0.5) + 200 * poulet"
            *   `Sport`: "15*8+ 2*WALKING_CALORIES(7.5,WEIGHT,750,19) + RUNNING_CALORIES(10,WEIGHT,1200,25)"
        *   **Row 2 (Simple Food, No Sport):**
            *   `Nourriture`: "100 * farine + 50 * sucre"
            *   `Sport`: ""
        *   **Row 3 (Very Complex Formula):**
            *   `Nourriture`: "10 * seven_up_mojito + 0.682* (pain complet*0.8 + poulet*0.2)"
            *   `Sport`: "WALKING_CALORIES(5,WEIGHT,3600,0)"

*   **Test Execution & Assertions:**
    1.  **Run Processing:** The main function of `build_features.py` will be called on these dummy files.
    2.  **Column Formatting:** Assert that all nutrient column names in the resulting DataFrame are correctly formatted (e.g., `Calories / 100g` becomes `calories`, `Free sugar` becomes `free_sugar`).
    3.  **Nutrient Calculation Verification:** For each row in the dummy journal, manually pre-calculate the expected value for **every single nutrient** based on the dummy `variables.csv`. The test will then iterate through each row and each nutrient column, asserting that the processed value is approximately equal to the pre-calculated expected value (`np.isclose` will be used to handle floating-point inaccuracies).
    4.  **Sport Calorie Verification:** For each row, manually pre-calculate the expected calorie expenditure from the `Sport` formula. The test will assert that the `sport` column in the output DataFrame matches this expected value.

**Definition of Done:**
*   The `build_features.py` script is implemented as specified.
*   The `tests/test_build_features.py` script exists and all tests pass.
*   The script, when run, generates a `data/features.csv` file that conforms to the specified schema.