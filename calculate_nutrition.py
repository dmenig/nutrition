from nutrition_calculator import calculate_nutrient_from_formula
import subprocess
import pandas as pd
import os


def main():
    """
    Main function to demonstrate on-the-fly nutrition calculation on processed journal data.
    """
    # Ensure data/processed_journal.csv is created
    print(
        "Running process_nutrition_journal.py to ensure data/processed_journal.csv exists..."
    )
    script_path = "process_nutrition_journal.py"
    try:
        subprocess.run(
            ["python3", script_path], check=True, capture_output=True, text=True
        )
        print("process_nutrition_journal.py executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running process_nutrition_journal.py: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return

    # Read the processed_journal.csv file
    processed_journal_path = os.path.join("data", "processed_journal.csv")
    if not os.path.exists(processed_journal_path):
        print(
            f"Error: {processed_journal_path} not found after running processing script."
        )
        return

    try:
        journal_df = pd.read_csv(processed_journal_path)
        print(f"Successfully loaded {processed_journal_path}")
    except Exception as e:
        print(f"Error reading {processed_journal_path}: {e}")
        return

    print("\nCalculating Kcal for the first 5 entries in processed_journal.csv:")
    # Iterate through the first 5 rows and calculate Kcal
    for index, row in journal_df.head(5).iterrows():
        formula = row["Nourriture"]
        if pd.isna(formula):
            print(
                f"Row {index + 1}: No formula found in 'Nourriture' column. Skipping."
            )
            continue

        try:
            calculated_kcal = calculate_nutrient_from_formula(
                str(formula), "Calories / 100g"
            )
            print(f"Formula: '{formula}' -> Kcal: {calculated_kcal:.2f}")
        except ValueError as e:
            print(f"Error calculating Kcal for formula '{formula}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred for formula '{formula}': {e}")


if __name__ == "__main__":
    main()
