import argparse
import os
import re
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
from sqlalchemy import create_engine, text

from app.core.config import settings
from nutrition_calculator import (
    get_nutrient_context,
    calculate_nutrient_from_formula_with_context,
)
from utils import normalize_food_names


NUTRIENT_COLUMN_TO_DB_FIELD = {
    "Calories / 100g": "calories",
    "Protéine": "protein",
    "Carbs": "carbs",
    "Fat": "fat",
}


def load_inputs(
    journal_path: str = "data/processed_journal.csv",
    variables_path: str = "data/variables.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(journal_path):
        raise FileNotFoundError(f"Missing journal CSV: {journal_path}")
    if not os.path.exists(variables_path):
        raise FileNotFoundError(f"Missing variables CSV: {variables_path}")
    journal_df = pd.read_csv(journal_path)
    variables_df = pd.read_csv(variables_path)
    journal_df["Date"] = pd.to_datetime(journal_df["Date"]).dt.date
    return journal_df, variables_df


def normalize_formula_identifiers(formula: str, variables_df: pd.DataFrame) -> str:
    if not isinstance(formula, str) or not formula.strip():
        return ""
    # Build mapping from original names to normalized identifiers
    name_map: Dict[str, str] = {
        str(row["Nom"]): normalize_food_names(row["Nom"]) for _, row in variables_df.iterrows()
    }
    # Replace with whole-word boundaries, case-insensitive
    normalized_formula = formula
    for original_name, normalized_name in name_map.items():
        if not original_name:
            continue
        normalized_formula = re.sub(
            rf"\b{re.escape(original_name)}\b", normalized_name, normalized_formula, flags=re.IGNORECASE
        )
    # Decimal commas to dots
    normalized_formula = normalized_formula.replace(",", ".")
    return normalized_formula


def compute_csv_totals_for_date(
    target_date: datetime.date, journal_df: pd.DataFrame, variables_df: pd.DataFrame
) -> Dict[str, float]:
    row = journal_df[journal_df["Date"] == target_date]
    if row.empty:
        raise ValueError(f"No journal entry for date: {target_date}")
    nourriture_formula = str(row.iloc[0]["Nourriture"]) if "Nourriture" in row.columns else ""
    if not isinstance(nourriture_formula, str) or not nourriture_formula.strip():
        return {db_field: 0.0 for db_field in NUTRIENT_COLUMN_TO_DB_FIELD.values()}

    normalized_formula = normalize_formula_identifiers(nourriture_formula, variables_df)

    totals: Dict[str, float] = {}
    for variable_col, db_field in NUTRIENT_COLUMN_TO_DB_FIELD.items():
        # Build nutrient context using normalized food names
        context = get_nutrient_context(variable_col, variables_df)
        # Scale per-100g values down to per-gram to align with DB logic
        context_per_gram = {k: (v / 100.0) for k, v in context.items()}
        value = calculate_nutrient_from_formula_with_context(normalized_formula, context_per_gram)
        totals[db_field] = float(value)
    return totals


def query_db_totals_for_date(target_date: datetime.date, user_email: Optional[str] = None) -> Dict[str, Optional[float]]:
    engine = create_engine(settings.DATABASE_URL)
    date_str = target_date.strftime("%Y-%m-%d")

    where_user = ""
    params: Dict[str, str] = {"date": date_str}
    if user_email:
        where_user = "AND u.email = :email"
        params["email"] = user_email

    sql = text(
        (
            "SELECT "
            " COALESCE(SUM(fl.calories), 0) AS calories,"
            " COALESCE(SUM(fl.protein), 0)  AS protein,"
            " COALESCE(SUM(fl.carbs), 0)    AS carbs,"
            " COALESCE(SUM(fl.fat), 0)      AS fat"
            " FROM food_logs fl"
            " JOIN users u ON u.id = fl.user_id"
            " WHERE DATE(fl.logged_at AT TIME ZONE 'UTC') = :date "
            + where_user
        )
    )

    with engine.connect() as conn:
        row = conn.execute(sql, params).mappings().first()
        if not row:
            return {field: None for field in NUTRIENT_COLUMN_TO_DB_FIELD.values()}
        return {k: (float(row[k]) if row[k] is not None else None) for k in NUTRIENT_COLUMN_TO_DB_FIELD.values()}


def pick_default_date(journal_df: pd.DataFrame) -> datetime.date:
    if journal_df.empty:
        raise ValueError("Journal CSV is empty")
    return max(journal_df["Date"])  # last available date


def main():
    parser = argparse.ArgumentParser(description="Verify a day's totals against Neon DB")
    parser.add_argument("--date", help="Date to verify (YYYY-MM-DD)")
    parser.add_argument(
        "--email",
        help="User email to filter DB rows (defaults to 'dummy@example.com')",
        default="dummy@example.com",
    )
    parser.add_argument(
        "--journal",
        default="data/processed_journal.csv",
        help="Path to processed journal CSV",
    )
    parser.add_argument(
        "--variables",
        default="data/variables.csv",
        help="Path to variables CSV",
    )
    args = parser.parse_args()

    journal_df, variables_df = load_inputs(args.journal, args.variables)
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = pick_default_date(journal_df)

    csv_totals = compute_csv_totals_for_date(target_date, journal_df, variables_df)
    db_totals = query_db_totals_for_date(target_date, user_email=args.email)

    print(f"Date: {target_date}")
    print("CSV totals (per-gram scaled):", {k: round(v, 2) for k, v in csv_totals.items()})
    print("DB totals:", {k: (round(v, 2) if v is not None else None) for k, v in db_totals.items()})

    # Compare with a small tolerance
    tolerance = 1e-2
    mismatches = {}
    for key in NUTRIENT_COLUMN_TO_DB_FIELD.values():
        csv_val = csv_totals.get(key)
        db_val = db_totals.get(key)
        if db_val is None:
            mismatches[key] = (csv_val, db_val)
            continue
        if abs(csv_val - db_val) > tolerance:
            mismatches[key] = (csv_val, db_val)

    if mismatches:
        print("Mismatch detected:")
        for k, (csv_v, db_v) in mismatches.items():
            print(f"  {k}: CSV={csv_v:.4f} vs DB={(db_v if db_v is not None else 'None')}")
        exit(2)
    else:
        print("OK: CSV and DB totals match within tolerance.")


if __name__ == "__main__":
    main()


