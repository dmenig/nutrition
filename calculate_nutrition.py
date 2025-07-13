import pandas as pd
import re
import unicodedata
import ast
import operator as op

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

    # Correct specific food names before cleaning
    if food_name.lower() == "fromage":
        food_name = "fromage blanc"

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
        # Handle malformed entries like '0,114 *'
        if len(components) > len(ops) and not components[-1]:
            print(
                f"Warning: Trailing operator '{ops[-1]}' in '{expression_str}' will be ignored."
            )
            components.pop()
            ops.pop()

        # If after cleaning, there are no operators, parse the single component.
        if not ops:
            return parse_nourriture_recursive(components[0])

        # Handle 'qty * food * qty * food ...' pattern
        if all(op == "*" for op in ops) and len(components) > 1:
            is_pattern = True
            # Check for alternating number/food pattern, must be even number of components
            if len(components) % 2 != 0:
                is_pattern = False
            else:
                for i, component in enumerate(components):
                    is_number = False
                    try:
                        eval_expr(component)
                        is_number = True
                    except (TypeError, SyntaxError, ValueError):
                        pass  # It's not a number, so it could be a food

                    if i % 2 == 0:  # Even indices must be numbers
                        if not is_number:
                            is_pattern = False
                            break
                    else:  # Odd indices must be foods
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

        # Fallback to original left-to-right evaluation for other cases
        left_res = parse_nourriture_recursive(components[0])
        is_numeric = not left_res
        if is_numeric:
            left_val = eval_expr(components[0])

        for i, op_str in enumerate(ops):
            right_comp = components[i + 1]
            right_res = parse_nourriture_recursive(right_comp)

            if not right_res:  # Right side is a number
                right_val = eval_expr(right_comp)
                if is_numeric:  # number op number
                    left_val = (
                        op.mul(left_val, right_val)
                        if op_str == "*"
                        else op.truediv(left_val, right_val)
                    )
                else:  # food op number
                    for food in left_res:
                        left_res[food] = (
                            op.mul(left_res[food], right_val)
                            if op_str == "*"
                            else op.truediv(left_res[food], right_val)
                        )
            else:  # Right side is a food
                if is_numeric:  # number op food
                    if op_str == "*":
                        for food in right_res:
                            right_res[food] *= left_val
                        left_res = right_res
                        is_numeric = False
                    else:  # op == '/'
                        raise ValueError(
                            f"Invalid operation in '{expression_str}': cannot divide a number by a food."
                        )
                else:  # food op food (AMBIGUOUS)
                    print(
                        f"Warning: Ambiguous operation between foods in '{expression_str}'. Interpreting as a sum."
                    )
                    for food, qty in right_res.items():
                        left_res[food] = left_res.get(food, 0) + qty
        return left_res if not is_numeric else {}

    # 3. Base case: must be a food name or a number
    try:
        eval_expr(expression_str)
        return {}  # It's a number
    except (TypeError, SyntaxError, ValueError):
        food_name = expression_str.strip()
        if food_name.lower() == "fromage":
            food_name = "fromage blanc"
        if not food_name:
            return {}  # Ignore empty parts
        # We can't validate with get_clean_food_name here as it would create a loop
        return {food_name: 1.0}  # It's a food


# 1. Read Nutritional Data
try:
    nutritional_data = pd.read_csv("data/variables.csv")
except FileNotFoundError:
    print("Error: 'data/variables.csv' not found.")
    exit()

nutritional_data["key"] = nutritional_data["Nom"].apply(get_clean_food_name)
nutritional_data.drop_duplicates(subset="key", keep="first", inplace=True)
nutrition_dict = nutritional_data.set_index("key").to_dict("index")

# 2. Process Journal Data
try:
    journal_df = pd.read_csv("data/processed_journal.csv")
except FileNotFoundError:
    print("Error: 'data/processed_journal.csv' not found.")
    exit()

results = []

# 3. Parse and Calculate
for index, row in journal_df.iterrows():
    nourriture_str = row["Nourriture"]
    daily_totals = {
        col: 0 for col in nutritional_data.columns if col not in ["Nom", "key"]
    }

    if pd.notna(nourriture_str):
        try:
            # Pre-process to add spaces around operators to help splitting
            processed_str = re.sub(r"([+*/\(\)-])", r" \1 ", str(nourriture_str))
            processed_str = re.sub(r"\s+", " ", processed_str).strip()
            food_quantities = parse_nourriture_recursive(processed_str)
            for food, quantity in food_quantities.items():
                add_food_to_totals(food, quantity, daily_totals, nutrition_dict)
        except (ValueError, TypeError) as e:
            print(f"Could not parse '{nourriture_str}' on row {index + 2}: {e}")

    result_row = row.to_dict()
    for nutrient, total_value in daily_totals.items():
        result_row[f"Total {nutrient.replace(' / 100g', '')}"] = total_value
    results.append(result_row)

# 4. Output Results
results_df = pd.DataFrame(results)
original_cols = list(journal_df.columns)
calculated_cols = [col for col in results_df.columns if col not in original_cols]
results_df = results_df[original_cols + calculated_cols]

# 5. Save Results
results_df.to_csv("data/calculated_nutrition.csv", index=False)

print(
    "Nutritional calculation complete. Results saved to 'data/calculated_nutrition.csv'"
)
