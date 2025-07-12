import os
import pandas as pd
import openpyxl
from datetime import datetime, timedelta
import glob


def process_nutrition_journal():
    """
    Processes a nutrition journal from an Excel file.
    1. Finds the first .xlsx file in the current directory.
    2. Processes the "Journal" and "Variables" sheets.
    3. In the "Journal" sheet, it reconstructs dates by incrementing from the first valid date.
    4. Filters the "Journal" data for dates on or after 2024-06-30.
    5. Saves the processed "Journal" data to 'processed_journal.csv'.
    6. Saves the "Variables" data to 'variables.csv'.
    """
    # --- 1. Find the first .xlsx file ---
    data_dir = "data"
    xlsx_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    if not xlsx_files:
        print(f"Error: No .xlsx file found in the '{data_dir}' directory.")
        return
    source_file = xlsx_files[0]

    # --- 2. Process "Journal" sheet ---
    try:
        workbook = openpyxl.load_workbook(source_file, data_only=False)
        sheet = workbook["Journal"]

        header = [cell.value for cell in sheet[1]]
        date_col_idx = header.index("Date")
        pds_col_idx = header.index("Pds")
        nourriture_col_idx = header.index("Nourriture")
        sport_col_idx = header.index("Sport")

        data = []
        current_date = None
        first_date_found = False

        # --- 3. Reconstruct dates with increment ---
        for row_cells in sheet.iter_rows(min_row=2):
            if not first_date_found:
                date_cell_value = row_cells[date_col_idx].value
                if date_cell_value is not None and isinstance(
                    date_cell_value, datetime
                ):
                    current_date = date_cell_value.date()
                    first_date_found = True
            elif current_date:
                current_date += timedelta(days=1)

            data.append(
                {
                    "Date": current_date,
                    "Pds": row_cells[pds_col_idx].value,
                    "Nourriture": row_cells[nourriture_col_idx].value,
                    "Sport": row_cells[sport_col_idx].value,
                }
            )

        journal_df = pd.DataFrame(data)
        journal_df.dropna(subset=["Date"], inplace=True)
        journal_df["Date"] = pd.to_datetime(journal_df["Date"])

        # --- 4. Filter "Journal" data ---
        filtered_journal_df = journal_df[journal_df["Date"] >= "2024-06-30"].copy()

        # --- 5. Save processed "Journal" data ---
        filtered_journal_df.to_csv("processed_journal.csv", index=False)
        print(
            "Successfully processed 'Journal' sheet and saved to processed_journal.csv"
        )

    except (KeyError, ValueError) as e:
        print(
            f"Error processing 'Journal' sheet: Required sheet or column not found. Details: {e}"
        )
        return
    except Exception as e:
        print(f"An unexpected error occurred while processing the 'Journal' sheet: {e}")
        return

    # --- 6. Process and save "Variables" sheet ---
    try:
        variables_df = pd.read_excel(source_file, sheet_name="Variables")
        variables_df.to_csv("variables.csv", index=False)
        print("Successfully processed 'Variables' sheet and saved to variables.csv")
    except Exception as e:
        print(f"Could not process 'Variables' sheet: {e}")


if __name__ == "__main__":
    process_nutrition_journal()
