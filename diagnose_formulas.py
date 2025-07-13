import pandas as pd
import re
import unicodedata
import ast
import operator as op

# --- Functions from calculate_nutrition.py ---

# --- Safe Expression Evaluation ---
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def eval_expr(expr):
    """Safely evaluates a mathematical string expression."""
    return eval_(ast.parse(expr.strip().replace(",", "."), mode="eval").body)


def eval_(node):
    """Recursively evaluates an AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Num):  # For older Python versions
        return node.n
    elif isinstance(node, ast.BinOp):
        return operators[type(node.op)](eval_(node.left), eval_(node.right))
    elif isinstance(node, ast.UnaryOp):
        return operators[type(node.op)](eval_(node.operand))
    else:
        raise TypeError(node)


def strip_accents(text):
    """Removes accents from a string."""
    if not isinstance(text, str):
        return text
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def get_clean_food_name(food_name):
    """Cleans a food name for dictionary lookup."""
    if not isinstance(food_name, str):
        return ""
    if "salade_boulgour_quinoa_fruits_secsm" in food_name:
        food_name = "salade_boulgour_quinoa_fruits_secs"

    cleaned_name = strip_accents(food_name.lower())
    cleaned_name = re.sub(r"[^a-z0-9_]+", "_", cleaned_name)
    cleaned_name = re.sub(r"_+", "_", cleaned_name)
    return cleaned_name.strip("_")


def add_food_to_totals(food_name, quantity, daily_totals, nutrition_dict):
    """Looks up a food and adds its nutritional values to the daily totals."""
    cleaned_name = get_clean_food_name(food_name)

    if not cleaned_name:
        return

    if cleaned_name in nutrition_dict:
        food_nutrients = nutrition_dict[cleaned_name]
        for nutrient, value in food_nutrients.items():
            if nutrient not in ["Nom", "key"] and pd.notna(value):
                daily_totals[nutrient] += (quantity / 100) * float(value)
    else:
        if cleaned_name not in ["prot", "eau"]:
            raise ValueError(
                f"Food not found: '{food_name}' (cleaned: '{cleaned_name}')"
            )


def parse_nourriture_recursive(expression_str):
    """
    Recursively parses a food expression string and returns a dictionary
    of {food_name: quantity}.
    """
    expression_str = expression_str.strip()
    if not expression_str:
        return {}

    # Handle parentheses first to determine the structure
    if expression_str.startswith("(") and expression_str.endswith(")"):
        balance = 0
        is_fully_enclosed = True
        for i, char in enumerate(expression_str):
            if char == "(":
                balance += 1
            elif char == ")":
                balance -= 1
            if balance == 0 and i < len(expression_str) - 1:
                is_fully_enclosed = False
                break
        if is_fully_enclosed:
            return parse_nourriture_recursive(expression_str[1:-1])

    # 1. Split by '+' or '-' outside parentheses (lowest precedence)
    balance = 0
    for i in range(len(expression_str) - 1, -1, -1):
        char = expression_str[i]
        if char == ")":
            balance += 1
        elif char == "(":
            balance -= 1
        elif char in "+-" and balance == 0:
            if i > 0 and expression_str[i - 1].lower() in "*/(e":
                continue
            operation = char
            left = expression_str[:i].strip()
            right = expression_str[i + 1 :].strip()
            left_foods = parse_nourriture_recursive(left)
            right_foods = parse_nourriture_recursive(right)
            for food, qty in right_foods.items():
                if operation == "+":
                    left_foods[food] = left_foods.get(food, 0) + qty
                else:
                    left_foods[food] = left_foods.get(food, 0) - qty
            return left_foods

    # 2. Handle '*' and '/' iteratively (left-associative)
    balance = 0
    components = []
    ops = []
    last_split = 0
    for i, char in enumerate(expression_str):
        if char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
        elif char in "*/" and balance == 0:
            components.append(expression_str[last_split:i].strip())
            ops.append(char)
            last_split = i + 1
    components.append(expression_str[last_split:].strip())

    if ops:
        if len(components) > len(ops) and not components[-1]:
            print(
                f"Warning: Trailing operator '{ops[-1]}' in '{expression_str}' will be ignored."
            )
            components.pop()
            ops.pop()

        if not ops:
            return parse_nourriture_recursive(components[0])

        if all(op == "*" for op in ops) and len(components) > 1:
            is_pattern = True
            if len(components) % 2 != 0:
                is_pattern = False
            else:
                for i, component in enumerate(components):
                    is_number = False
                    try:
                        eval_expr(component)
                        is_number = True
                    except (TypeError, SyntaxError, ValueError):
                        pass

                    if i % 2 == 0:
                        if not is_number:
                            is_pattern = False
                            break
                    else:
                        if is_number:
                            is_pattern = False
                            break

            if is_pattern:
                total_foods = {}
                for i in range(0, len(components), 2):
                    quantity = eval_expr(components[i])
                    food_dict = parse_nourriture_recursive(components[i + 1])
                    for food, base_qty in food_dict.items():
                        total_foods[food] = (
                            total_foods.get(food, 0) + quantity * base_qty
                        )
                return total_foods

        left_res = parse_nourriture_recursive(components[0])
        is_numeric = not left_res
        if is_numeric:
            left_val = eval_expr(components[0])

        for i, op_str in enumerate(ops):
            right_comp = components[i + 1]
            right_res = parse_nourriture_recursive(right_comp)

            if not right_res:
                right_val = eval_expr(right_comp)
                if is_numeric:
                    left_val = (
                        op.mul(left_val, right_val)
                        if op_str == "*"
                        else op.truediv(left_val, right_val)
                    )
                else:
                    for food in left_res:
                        left_res[food] = (
                            op.mul(left_res[food], right_val)
                            if op_str == "*"
                            else op.truediv(left_res[food], right_val)
                        )
            else:
                if is_numeric:
                    if op_str == "*":
                        for food in right_res:
                            right_res[food] *= left_val
                        left_res = right_res
                        is_numeric = False
                    else:
                        raise ValueError(
                            f"Invalid operation in '{expression_str}': cannot divide a number by a food."
                        )
                else:
                    print(
                        f"Warning: Ambiguous operation between foods in '{expression_str}'. Interpreting as a sum."
                    )
                    for food, qty in right_res.items():
                        left_res[food] = left_res.get(food, 0) + qty
        return left_res if not is_numeric else {}

    # 3. Base case: must be a food name or a number
    try:
        eval_expr(expression_str)
        return {}
    except (TypeError, SyntaxError, ValueError):
        food_name = expression_str.strip()
        if not food_name:
            return {}
        if not get_clean_food_name(food_name):
            raise ValueError(f"Malformed entry found: invalid food name '{food_name}'")
        return {food_name: 1.0}


# --- Diagnostic Script ---


def run_diagnostics():
    """Runs diagnostic tests on a set of food formulas."""
    # 1. Load Nutritional Data
    try:
        nutritional_data = pd.read_csv("data/variables.csv")
    except FileNotFoundError:
        print("Error: 'data/variables.csv' not found.")
        return

    nutritional_data["key"] = nutritional_data["Nom"].apply(get_clean_food_name)
    nutritional_data.drop_duplicates(subset="key", keep="first", inplace=True)
    nutrition_dict = nutritional_data.set_index("key").to_dict("index")

    # 2. Define Example Formulas
    example_formulas = [
        "100*oeuf+50*fromage blanc",
        "5.61 / 5 * kohlrouladen",
        "0,5 * tomate * 0,4 * chou_fleur",
        "150 * (pates + sauce_tomate)",
    ]

    # 3. Process and Print Results
    for formula in example_formulas:
        print(f"Original Formula: {formula}")

        try:
            # Pre-process to add spaces around operators to help splitting
            processed_str = re.sub(r"([+*/\(\)-])", r" \1 ", str(formula))
            processed_str = re.sub(r"\s+", " ", processed_str).strip()

            parsed_representation = parse_nourriture_recursive(processed_str)
            print(f"Parsed Representation: {parsed_representation}")

            # Calculate totals
            daily_totals = {
                col: 0 for col in nutritional_data.columns if col not in ["Nom", "key"]
            }
            for food, quantity in parsed_representation.items():
                add_food_to_totals(food, quantity, daily_totals, nutrition_dict)

            # --- Detailed Breakdown for Specific Formula ---
            if formula == "150 * (pates + sauce_tomate)":
                print("\n--- Detailed Calorie Calculation Breakdown ---")

                # a. Fetch and print calorie value for 'pates'
                pates_key = get_clean_food_name("pates")
                calories_pates = nutrition_dict.get(pates_key, {}).get(
                    "Calories / 100g", "Not Found"
                )
                print(f"1. Calories for 'pates': {calories_pates} / 100g")

                # b. Fetch and print calorie value for 'sauce_tomate'
                sauce_key = get_clean_food_name("sauce_tomate")
                calories_sauce = nutrition_dict.get(sauce_key, {}).get(
                    "Calories / 100g", "Not Found"
                )
                print(f"2. Calories for 'sauce_tomate': {calories_sauce} / 100g")

                # c. Explain the formula interpretation
                print(
                    "3. Formula Interpretation: '150 * (pates + sauce_tomate)' becomes '(1.5 * calories_pates) + (1.5 * calories_sauce_tomate)'"
                )

                # d. Show the final calculation with actual numbers
                if isinstance(calories_pates, (int, float)) and isinstance(
                    calories_sauce, (int, float)
                ):
                    result = (1.5 * calories_pates) + (1.5 * calories_sauce)
                    print(
                        f"4. Final Calculation: (1.5 * {calories_pates}) + (1.5 * {calories_sauce}) = {result:.2f}\n"
                    )
                else:
                    print(
                        "Could not perform final calculation due to missing calorie data.\n"
                    )

            # Format and print calculated nutrients
            calculated_nutrients = {
                f"Total {k.replace(' / 100g', '')}": round(v, 2)
                for k, v in daily_totals.items()
                if v > 0
            }
            print(f"Final Calculated Nutrients: {calculated_nutrients}")

        except (ValueError, TypeError, FileNotFoundError) as e:
            print(f"Error processing formula: {e}")

        print("---")


if __name__ == "__main__":
    run_diagnostics()
