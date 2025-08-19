import csv
import os
import pandas as pd
import re
import ast
from pathlib import Path
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
from typing import List, Tuple, Optional
from utils import SafeFormulaEvaluator, SafeSportFormulaEvaluator, normalize_food_names
from app.db.models import Base, Food, FoodLog, User, SportActivity, WeightLog
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _resolve_data_file(name: str) -> str:
    """Return an absolute path to a data file that exists across local/dev/prod.

    Tries common locations in this order:
    - /app/data (inside container)
    - <repo_or_app_root>/data relative to this file
    - CWD/data (fallback)
    """
    candidates = [
        Path("/app/data") / name,
        Path(__file__).resolve().parents[2] / "data" / name,
        Path.cwd() / "data" / name,
    ]
    for p in candidates:
        try:
            if p.exists():
                return str(p)
        except Exception:
            continue
    return str(Path("data") / name)


def populate_food_table():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Read and insert new entries
        with open(_resolve_data_file("variables.csv"), mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Check if food already exists
                existing_food = db.query(Food).filter(Food.name == row["Nom"]).first()
                if existing_food:
                    # Update existing food
                    existing_food.calories = (
                        float(row["Calories / 100g"]) if row["Calories / 100g"] else 0
                    )
                    existing_food.protein = (
                        float(row["Protéine"]) if row["Protéine"] else 0
                    )
                    existing_food.carbs = float(row["Carbs"]) if row["Carbs"] else 0
                    existing_food.fat = float(row["Fat"]) if row["Fat"] else 0
                    # Extended nutrients
                    existing_food.sugar = float(row.get("Sugar") or 0)
                    existing_food.sel = float(row.get("Sel") or 0)
                    existing_food.alcool = float(row.get("Alcool") or 0)
                    existing_food.water = float(row.get("Water") or 0)
                    db.commit()
                else:
                    # Create new food
                    food = Food(
                        name=row["Nom"],
                        calories=float(row["Calories / 100g"])
                        if row["Calories / 100g"]
                        else 0,
                        protein=float(row["Protéine"]) if row["Protéine"] else 0,
                        carbs=float(row["Carbs"]) if row["Carbs"] else 0,
                        fat=float(row["Fat"]) if row["Fat"] else 0,
                        sugar=float(row.get("Sugar") or 0),
                        sel=float(row.get("Sel") or 0),
                        alcool=float(row.get("Alcool") or 0),
                        water=float(row.get("Water") or 0),
                    )
                    db.add(food)
                    db.commit()
            print("Food table populated successfully!")
    finally:
        db.close()


def parse_food_formula(formula: str) -> list:
    """
    Parses a food formula string and returns a list of (quantity, food_name) tuples.

    Args:
        formula: A string representing the food formula (e.g., "100 * pain + 50 * fromage").

    Returns:
        A list of (quantity, food_name) tuples.
    """
    # Replace commas with dots for decimal compatibility
    formula = formula.replace(",", ".")

    # Parse the expression into an AST
    try:
        node = ast.parse(formula, mode="eval")
    except SyntaxError:
        return []

    # Extract food items and quantities from the AST
    food_items = []
    _extract_food_items(node.body, food_items)
    return food_items


def _extract_food_items(node, food_items, current_quantity=1.0):
    """
    Recursively extracts food items and their quantities from an AST node.
    """
    # Helper to evaluate purely numeric sub-expressions (no variable names)
    def _eval_numeric(subnode):
        try:
            evaluator = SafeFormulaEvaluator()
            return float(evaluator.visit(subnode))
        except Exception:
            return None

    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Mult, ast.Div)):
            # Try to evaluate either side if it's a purely numeric expression
            left_numeric = _eval_numeric(node.left)
            right_numeric = _eval_numeric(node.right)

            if isinstance(node.op, ast.Mult):
                if left_numeric is not None:
                    _extract_food_items(node.right, food_items, current_quantity * left_numeric)
                    return
                if right_numeric is not None:
                    _extract_food_items(node.left, food_items, current_quantity * right_numeric)
                    return
                # Fallback: propagate through both sides (rare ambiguous case)
                _extract_food_items(node.left, food_items, current_quantity)
                _extract_food_items(node.right, food_items, current_quantity)
                return

            # Division
            if right_numeric is not None:
                _extract_food_items(node.left, food_items, current_quantity / right_numeric)
                return
            if left_numeric is not None and isinstance(node.right, ast.Name):
                # Case like: numeric / name → interpret as (numeric) distributed to name denominator
                # Not expected in our data; skip to avoid incorrect attribution.
                return
            # Fallback
            _extract_food_items(node.left, food_items, current_quantity)
            _extract_food_items(node.right, food_items, current_quantity)
        elif isinstance(node.op, ast.Add):
            # Handle addition (food_item + food_item)
            _extract_food_items(node.left, food_items, current_quantity)
            _extract_food_items(node.right, food_items, current_quantity)
        elif isinstance(node.op, ast.Sub):
            # Handle subtraction (food_item - food_item)
            _extract_food_items(node.left, food_items, current_quantity)
            _extract_food_items(node.right, food_items, -current_quantity)
    elif isinstance(node, ast.Name):
        # Handle standalone food name (quantity defaults to 1)
        food_items.append((current_quantity, node.id))
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        # Handle standalone number (ignore)
        pass
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        # Handle unary operations
        multiplier = -1 if isinstance(node.op, ast.USub) else 1
        _extract_food_items(node.operand, food_items, current_quantity * multiplier)


def populate_food_log_table():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Load the processed journal data
        journal_df = pd.read_csv(_resolve_data_file("processed_journal.csv"))
        variables_df = pd.read_csv(_resolve_data_file("variables.csv"))
        # We'll use processed_journal.csv "Pds" column as the canonical observed weight source

        # Create a mapping from original to normalized food names
        food_name_mapping = {
            row["Nom"]: normalize_food_names(row["Nom"])
            for _, row in variables_df.iterrows()
        }

        # Create a reverse mapping from normalized to original food names
        reverse_food_name_mapping = {
            normalize_food_names(row["Nom"]): row["Nom"]
            for _, row in variables_df.iterrows()
        }

        # Create a dictionary to store nutritional information for each food
        food_nutrition = {}
        for _, row in variables_df.iterrows():
            normalized_name = normalize_food_names(row["Nom"])
            food_nutrition[normalized_name] = {
                "calories": float(row["Calories / 100g"])
                if row["Calories / 100g"]
                else 0,
                "protein": float(row["Protéine"]) if row["Protéine"] else 0,
                "carbs": float(row["Carbs"]) if row["Carbs"] else 0,
                "fat": float(row["Fat"]) if row["Fat"] else 0,
            }

        # Create a dummy user for the logs
        dummy_user = db.query(User).filter(User.email == "dummy@example.com").first()
        if not dummy_user:
            # Create a dummy user with all required fields
            # Note: We're using a direct SQL insert here because the User model
            # in models.py doesn't match the actual database schema
            from sqlalchemy import text

            import uuid

            user_id = str(uuid.uuid4())
            result = db.execute(
                text("""
                INSERT INTO users (id, email, hashed_password, username)
                VALUES (:id, :email, :hashed_password, :username)
                RETURNING id
            """),
                {
                    "id": user_id,
                    "email": "dummy@example.com",
                    "hashed_password": "dummy_hashed_password",
                    "username": "dummy",
                },
            )
            db.commit()
            user_id = result.fetchone()[0]
            dummy_user = db.query(User).filter(User.id == user_id).first()

        default_user_id = dummy_user.id

        # Clear existing logs for dummy user before repopulating
        db.query(FoodLog).filter(FoodLog.user_id == default_user_id).delete()
        db.commit()
        # Optionally clear weight logs for dummy user
        db.query(WeightLog).filter(WeightLog.user_id == default_user_id).delete()
        db.commit()
        # Process each row in the journal
        for _, row in journal_df.iterrows():
            date = pd.to_datetime(row["Date"])
            food_formula = str(row["Nourriture"]) if pd.notna(row["Nourriture"]) else ""

            if not food_formula:
                continue

            # Normalize food names within the formula
            for original_name, normalized_name in food_name_mapping.items():
                # Use regex to replace whole words only, ignoring case
                food_formula = re.sub(
                    r"\b" + re.escape(original_name) + r"\b",
                    normalized_name,
                    food_formula,
                    flags=re.IGNORECASE,
                )

            # Parse the formula to extract food items and quantities
            food_items = parse_food_formula(food_formula)
            if not food_items:
                raise ValueError(f"No food items parsed for date {date} with formula '{food_formula}'")

            # Create a FoodLog entry for each food item
            for quantity, normalized_food_name in food_items:
                if quantity is None or float(quantity) == 0:
                    raise ValueError(
                        f"Zero quantity for food '{normalized_food_name}' on {date} in formula '{food_formula}'"
                    )
                # Get the original food name
                food_name = reverse_food_name_mapping.get(
                    normalized_food_name, normalized_food_name
                )

                # Get nutritional information for the food
                nutrition = food_nutrition.get(
                    normalized_food_name,
                    {
                        "calories": 0,
                        "protein": 0,
                        "carbs": 0,
                        "fat": 0,
                    },
                )

                # Interpret all parsed coefficients as counts of 100g servings.
                # This ensures expressions like "15*schweppes_zero" map to 1500g.
                quantity_in_grams = float(quantity) * 100.0

                # The nutritional values in the CSV are per 100g.
                # Scale by (grams / 100) to compute actual amounts.
                scale = quantity_in_grams / 100
                calories = nutrition["calories"] * scale
                protein = nutrition["protein"] * scale
                carbs = nutrition["carbs"] * scale
                fat = nutrition["fat"] * scale

                # Create a FoodLog entry
                food_log = FoodLog(
                    user_id=default_user_id,
                    food_name=food_name,
                    quantity=quantity_in_grams,
                    unit="g",
                    calories=calories,
                    protein=protein,
                    carbs=carbs,
                    fat=fat,
                    logged_at=date,
                )
                db.add(food_log)

        db.commit()
        # Insert observed weights from processed_journal.csv (Pds column)
        if "Pds" in journal_df.columns:
            for _, row in journal_df.iterrows():
                try:
                    date_ts = pd.to_datetime(row["Date"])  # pandas Timestamp (date-only)
                    weight_val = float(row.get("Pds") or 0)
                    if weight_val == 0:
                        continue
                except Exception:
                    continue
                from datetime import datetime as _dt, timezone as _tz
                date_dt = _dt(date_ts.year, date_ts.month, date_ts.day, tzinfo=_tz.utc)
                wl = WeightLog(
                    user_id=default_user_id,
                    weight_kg=weight_val,
                    logged_at=date_dt,
                    logged_date=date_dt.date(),
                )
                db.add(wl)
            db.commit()
        print("FoodLog table populated successfully!")
    finally:
        db.close()


def verify_population():
    with engine.connect() as connection:
        food_count = connection.execute(text("SELECT COUNT(*) FROM foods")).scalar_one()
        food_log_count = connection.execute(
            text("SELECT COUNT(*) FROM food_logs")
        ).scalar_one()
        sport_count = connection.execute(
            text("SELECT COUNT(*) FROM sport_activities")
        ).scalar_one()
        print(f"Rows in foods: {food_count}")
        print(f"Rows in food_logs: {food_log_count}")
        print(f"Rows in sport_activities: {sport_count}")

def _extract_sport_calls_with_calories(expr_eval: str) -> List[Tuple[str, int, float, Optional[float], Optional[float]]]:
    """
    Finds every whitelisted sport function call in the expression and returns a list of
    tuples: (activity_name, duration_minutes, calories_for_that_call, carried_weight_kg, distance_m).

    Notes:
    - Assumes the expression already has WEIGHT substituted with a numeric value.
    - If a call's duration cannot be inferred, 0 is used.
    - If evaluation fails for a call, its calories default to 0.0.
    """
    import ast
    from sport_formulas import SPORT_FUNCTIONS

    def humanize(name_upper: str) -> str:
        mapping = {
            "WALKING_CALORIES": "Walking",
            "RUNNING_CALORIES": "Running",
            "CYCLING_CALORIES": "Cycling",
        }
        return mapping.get(name_upper, name_upper.title())

    try:
        root = ast.parse(expr_eval, mode="eval")
    except SyntaxError:
        return []

    # Case-insensitive function context to handle lower/upper cased formulas
    try:
        func_ctx = {**SPORT_FUNCTIONS}
        func_ctx.update({k.lower(): v for k, v in SPORT_FUNCTIONS.items()})
    except Exception:
        func_ctx = SPORT_FUNCTIONS
    evaluator = SafeSportFormulaEvaluator(func_ctx)
    calls: List[Tuple[str, int, float, Optional[float], Optional[float]]] = []

    def is_pure_numeric(node: ast.AST) -> bool:
        try:
            # Will raise if it requires a sport function
            evaluator.visit(node)
            return True
        except Exception:
            return False

    def has_call(node: ast.AST) -> bool:
        if isinstance(node, ast.Call):
            return True
        return any(has_call(child) for child in ast.iter_child_nodes(node))

    def extract_calls(node: ast.AST, factor: float = 1.0) -> None:
        # Direct call
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_upper = node.func.id.upper()
            activity_name = humanize(func_upper)
            duration_minutes = 0
            distance_m_val: float | None = None
            add_weight_val: float | None = None
            # Prefer keyword argument
            for kw in getattr(node, "keywords", []) or []:
                if kw.arg == "duration_minutes":
                    try:
                        duration_val = evaluator.visit(kw.value)
                        duration_minutes = int(round(float(duration_val)))
                    except Exception:
                        duration_minutes = 0
                    break
            # Fallback to first positional arg
            if duration_minutes == 0 and getattr(node, "args", []):
                try:
                    duration_val = evaluator.visit(node.args[0])
                    duration_minutes = int(round(float(duration_val)))
                except Exception:
                    duration_minutes = 0
            # Capture optional positional args if present
            try:
                if len(getattr(node, "args", [])) >= 3:
                    distance_m_val = float(evaluator.visit(node.args[2]))
                if len(getattr(node, "args", [])) >= 4:
                    add_weight_val = float(evaluator.visit(node.args[3]))
            except Exception:
                pass
            # Capture keyword overrides if present
            for kw in getattr(node, "keywords", []) or []:
                try:
                    if kw.arg == "distance_meters":
                        distance_m_val = float(evaluator.visit(kw.value))
                    if kw.arg == "additional_weight_kg":
                        add_weight_val = float(evaluator.visit(kw.value))
                except Exception:
                    pass
            # Base calories for this call (case-insensitive, robust evaluation)
            try:
                base_cal = float(evaluator.visit(node))
            except Exception:
                base_cal = 0.0

            # Propagate numeric multipliers/divisors to both calories and physical quantities
            # - Scale calories by the factor (can be fractional or negative)
            # - Scale duration and distance by abs(factor) so UI reflects repeated/combined effort
            scale_abs = abs(float(factor)) if isinstance(factor, (int, float)) else 1.0
            scaled_duration = int(round(duration_minutes * scale_abs)) if duration_minutes else 0
            scaled_distance = (
                float(distance_m_val) * scale_abs if distance_m_val is not None else None
            )
            scaled_calories = base_cal * float(factor)

            calls.append(
                (
                    activity_name,
                    scaled_duration,
                    scaled_calories,
                    add_weight_val,
                    scaled_distance,
                )
            )
            return

        # Binary operations can scale calls
        if isinstance(node, ast.BinOp):
            op = node.op
            left, right = node.left, node.right
            left_has = has_call(left)
            right_has = has_call(right)
            # Multiplication
            if isinstance(op, ast.Mult):
                if left_has and is_pure_numeric(right):
                    scale = float(evaluator.visit(right))
                    extract_calls(left, factor * scale)
                    return
                if right_has and is_pure_numeric(left):
                    scale = float(evaluator.visit(left))
                    extract_calls(right, factor * scale)
                    return
                # Both sides have calls or non-numeric scale → distribute
                extract_calls(left, factor)
                extract_calls(right, factor)
                return
            # Division
            if isinstance(op, ast.Div):
                if left_has and is_pure_numeric(right):
                    scale = float(evaluator.visit(right))
                    if scale != 0:
                        extract_calls(left, factor / scale)
                    else:
                        extract_calls(left, factor)
                    return
                # If call on right or non-numeric denominator, distribute
                extract_calls(left, factor)
                extract_calls(right, factor)
                return
            # Addition/Subtraction → no scaling, just traverse
            extract_calls(left, factor)
            if isinstance(op, ast.Sub):
                # Negative factor for right side
                extract_calls(right, -factor)
            else:
                extract_calls(right, factor)
            return

        # Unary +/- operators
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                extract_calls(node.operand, -factor)
            else:
                extract_calls(node.operand, factor)
            return

        # Generic traversal
        for child in ast.iter_child_nodes(node):
            extract_calls(child, factor)

    extract_calls(root.body, 1.0)
    return calls


def populate_sport_activities_table():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Ensure dummy user exists (reuse creation logic from food logs)
        dummy_user = db.query(User).filter(User.email == "dummy@example.com").first()
        if not dummy_user:
            from sqlalchemy import text
            import uuid
            user_id = str(uuid.uuid4())
            result = db.execute(
                text(
                    """
                INSERT INTO users (id, email, hashed_password, username)
                VALUES (:id, :email, :hashed_password, :username)
                RETURNING id
            """
                ),
                {
                    "id": user_id,
                    "email": "dummy@example.com",
                    "hashed_password": "dummy_hashed_password",
                    "username": "dummy",
                },
            )
            db.commit()
            user_id = result.fetchone()[0]
            dummy_user = db.query(User).filter(User.id == user_id).first()

        # Load journal data
        journal_df = pd.read_csv(_resolve_data_file("processed_journal.csv"))

        # Clear existing sports for dummy user before repopulating
        db.query(SportActivity).filter(SportActivity.user_id == dummy_user.id).delete()
        db.commit()

        from sport_formulas import evaluate_sport_formula

        inserted = 0
        for _, row in journal_df.iterrows():
            if pd.isna(row.get("Sport")) or str(row["Sport"]).strip() == "":
                continue
            # Normalize to UTC midnight to match API day filtering (UTC boundaries)
            from datetime import datetime, timezone as dt_timezone
            date_ts = pd.to_datetime(row["Date"])  # pandas Timestamp (date-only)
            date = datetime(
                date_ts.year, date_ts.month, date_ts.day, tzinfo=dt_timezone.utc
            )
            weight = float(row.get("Pds", 0) or 0)
            expr = str(row["Sport"]).lstrip("=").strip()
            # Normalize function names to uppercase to match whitelist (tolerate lowercase inputs)
            try:
                expr_norm = re.sub(r"([A-Za-z_]+)\(", lambda m: m.group(1).upper()+"(", expr)
            except Exception:
                expr_norm = expr
            # Substitute WEIGHT (any case) with actual value for safe evaluation
            expr_eval = re.sub(r"WEIGHT", str(weight), expr_norm, flags=re.IGNORECASE)

            # Split into individual calls so DB rows match the journal entries
            calls = _extract_sport_calls_with_calories(expr_eval)

            total_inserted_cals = 0.0
            for activity_name, duration_minutes, calories_expended, add_weight, distance_m in calls:
                sa = SportActivity(
                    user_id=dummy_user.id,
                    activity_name=activity_name,
                    duration_minutes=duration_minutes,
                    carried_weight_kg=add_weight,
                    distance_m=distance_m,
                    calories_expended=calories_expended,
                    logged_at=date,
                )
                db.add(sa)
                inserted += 1
                try:
                    total_inserted_cals += float(calories_expended)
                except Exception:
                    pass
            if inserted % 50 == 0:
                db.commit()
            # Correction: ensure DB daily sum matches direct evaluation of the expression
            try:
                evaluated_total = float(evaluate_sport_formula(expr_eval)) if expr_eval else 0.0
                diff = evaluated_total - total_inserted_cals
                if abs(diff) > 0.5:  # tolerate minor FP differences
                    sa_corr = SportActivity(
                        user_id=dummy_user.id,
                        activity_name="Correction",
                        duration_minutes=0,
                        carried_weight_kg=None,
                        distance_m=None,
                        calories_expended=diff,
                        logged_at=date,
                    )
                    db.add(sa_corr)
                    inserted += 1
            except Exception:
                # Ignore correction if evaluation fails
                pass
        db.commit()
        print(f"SportActivity table populated from journal CSV! Inserted {inserted} rows.")
    finally:
        db.close()


def main():
    """Populate foods, food logs, and sport activities, then print counts."""
    populate_food_table()
    populate_food_log_table()
    populate_sport_activities_table()
    verify_population()


# Direct script execution or `python -m app.db.populate_db` only
if __name__ == "__main__":
    main()
