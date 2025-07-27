import unicodedata
import ast
import operator as op
import pandas as pd




def strip_accents(text):
    """Removes accents from a string."""
    if not isinstance(text, str):
        return text
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


class SafeFormulaEvaluator(ast.NodeVisitor):
    """
    A safe AST node visitor to evaluate simple mathematical formulas.
    Supports basic arithmetic operations and numeric literals.
    """

    def __init__(self, context=None):
        self.context = context or {}
        self.operations = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.USub: op.neg,
            ast.UAdd: op.pos,
        }

    def visit_Num(self, node):
        """Handles numeric literals (deprecated in Python 3.8, use visit_Constant)."""
        return node.n

    def visit_Constant(self, node):
        """Handles numeric and string literals."""
        if isinstance(node.value, (int, float)):
            return node.value
        raise TypeError(f"Unsupported constant type: {type(node.value)}")

    def visit_BinOp(self, node):
        """Handles binary operations (+, -, *, /)."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        if pd.isna(left) or pd.isna(right):
            return pd.NA
        return self.operations[type(node.op)](left, right)

    def visit_UnaryOp(self, node):
        """Handles unary operations (e.g., negation)."""
        operand = self.visit(node.operand)
        return self.operations[type(node.op)](operand)

    def visit_Name(self, node):
        """Handles variable names."""
        # Normalize the variable name by replacing underscores with spaces
        if node.id == "WEIGHT":
            return self.context["WEIGHT"]
        normalized_id = strip_accents(node.id.lower()).replace(" ", "_").replace("'", "_")

        if normalized_id in self.context:
            return self.context[normalized_id]
        raise ValueError(f"Name '{node.id}' is not defined in the given context.")

    def visit_Call(self, node):
        """Disallow function calls for security."""
        raise ValueError("Function calls are not allowed in formulas.")


class SafeSportFormulaEvaluator(SafeFormulaEvaluator):
    """
    A safe AST node visitor for evaluating sport formulas.
    Inherits from SafeFormulaEvaluator and adds support for calling whitelisted functions.
    """

    def visit_Call(self, node):
        """Handles whitelisted function calls."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.context and callable(self.context[func_name]):
                args = [self.visit(arg) for arg in node.args]
                kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
                return self.context[func_name](*args, **kwargs)
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            raise NameError(f"Function '{func_name}' is not allowed.")
        raise NameError("Indirect function calls are not allowed.")
