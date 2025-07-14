# Part 1: Data Processing

**Goal:** Create a script that processes the raw data and produces a clean `features.csv` file.

**Files to Create/Modify:**
*   `data_processor.py`
*   `tests/test_data_processor.py`

**Implementation Details (`data_processor.py`):**
*   The `load_and_process_data` function will be updated to produce cleaner column names.
*   **New Logic for column names:** `strip_accents(nutrient.split(' / ')[0]).replace(" ", "_").lower()`
*   **Example:** `Calories / 100g` will become `calories`.
*
*   **Parsing Strategy:**
    *   **Nutrient Parsing:** The script will import and use the `calculate_nutrient_from_formula` function from `nutrition_calculator.py` to parse the `Nourriture` column for each nutrient. This leverages the existing, tested logic for handling complex food formulas.
    *   **Sport Parsing:** A new `SafeSportFormulaEvaluator` class will be created. This class will be adapted from the `SafeFormulaEvaluator` in `utils.py` but modified to allow function calls. It will use the `SPORT_FUNCTIONS` dictionary from `sport_formulas.py` as a strict whitelist to ensure only known, safe sport calculation functions can be executed from the `Sport` column formula. The `WEIGHT` variable in the formula will be dynamically replaced with the current day's weight (`pds` column).
*
*   **Error Handling:** The script will operate under a strict "no-fail" policy. All parsing functions must succeed. If any formula in the `Nourriture` or `Sport` columns cannot be parsed, the script will raise a `ValueError`. There will be no `try-except` blocks that default to `NaN` or `0` for failed parsing. This ensures data integrity and forces resolution of any data quality issues.
*
*   The script will output a pandas DataFrame to `data/features.csv` with the following schema:
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

**Testing (`tests/test_data_processor.py`):**

The test suite for `data_processor.py` will be comprehensive, using a single, powerful test function that covers multiple edge cases.

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
    1.  **Run Processing:** The `load_and_process_data` function will be called on these dummy files.
    2.  **Column Formatting:** Assert that all nutrient column names in the resulting DataFrame are correctly formatted (e.g., `Calories / 100g` becomes `calories`, `Free sugar` becomes `free_sugar`).
    3.  **Nutrient Calculation Verification:** For each row in the dummy journal, manually pre-calculate the expected value for **every single nutrient** based on the dummy `variables.csv`. The test will then iterate through each row and each nutrient column, asserting that the processed value is approximately equal to the pre-calculated expected value (`np.isclose` will be used to handle floating-point inaccuracies).
    4.  **Sport Calorie Verification:** For each row, manually pre-calculate the expected calorie expenditure from the `Sport` formula. The test will assert that the `sport` column in the output DataFrame matches this expected value.

**Definition of Done:**
*   The `data_processor.py` script is implemented as specified.
*   The `tests/test_data_processor.py` script exists and all tests pass.
*   The script, when run, generates a `data/features.csv` file that conforms to the specified schema.