import csv
import os
import pandas as pd
import re
import ast
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
from app.db.models import Base, Food, FoodLog, User
from app.core.config import settings
from utils import SafeFormulaEvaluator, normalize_food_names

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def populate_food_table():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Read and insert new entries
        with open("data/variables.csv", mode="r") as file:
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
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Mult, ast.Div)):
            # Handle multiplication or division (quantity * food_name or quantity / food_name)
            if isinstance(node.left, (ast.Constant, ast.Num)) and isinstance(
                node.right, ast.Name
            ):
                quantity = (
                    node.left.value
                    if isinstance(node.left, ast.Constant)
                    else node.left.n
                )
                if isinstance(node.op, ast.Div):
                    quantity = 1.0 / quantity
                food_items.append((quantity * current_quantity, node.right.id))
            elif isinstance(node.left, ast.Name) and isinstance(
                node.right, (ast.Constant, ast.Num)
            ):
                quantity = (
                    node.right.value
                    if isinstance(node.right, ast.Constant)
                    else node.right.n
                )
                if isinstance(node.op, ast.Div):
                    quantity = 1.0 / quantity
                food_items.append((quantity * current_quantity, node.left.id))
            else:
                # Recursively process both sides
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
        journal_df = pd.read_csv("data/processed_journal.csv")
        variables_df = pd.read_csv("data/variables.csv")

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

                # Interpret quantities:
                # - If quantity is a small number (<= 10), treat it as "number of 100g servings"
                #   to avoid defaulting bare items (e.g., "pain") to 1 gram.
                # - Otherwise assume it is already in grams.
                quantity_in_grams = quantity * 100 if quantity <= 10 else quantity

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
        print("FoodLog table populated successfully!")
    finally:
        db.close()


def verify_population():
    with engine.connect() as connection:
        food_count = connection.execute(text("SELECT COUNT(*) FROM foods")).scalar_one()
        food_log_count = connection.execute(
            text("SELECT COUNT(*) FROM food_logs")
        ).scalar_one()
        print(f"Rows in foods: {food_count}")
        print(f"Rows in food_logs: {food_log_count}")


if __name__ == "__main__":
    populate_food_table()
    populate_food_log_table()
    verify_population()
