import unicodedata


def strip_accents(text):
    """Removes accents from a string."""
    if not isinstance(text, str):
        return text
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


import ast
import operator as op


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
        }

    def visit_Num(self, node):
        """Handles numeric literals (deprecated in Python 3.8, use visit_Constant)."""
        return node.n

    def visit_Constant(self, node):
        """Handles numeric and string literals."""
        if isinstance(node.value, (int, float)):
            return node.value
        # For safety, you might want to restrict what kind of constants are allowed.
        # Here, we only allow numbers.
        raise TypeError(f"Unsupported constant type: {type(node.value)}")

    def visit_BinOp(self, node):
        """Handles binary operations (+, -, *, /)."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self.operations[type(node.op)](left, right)

    def visit_UnaryOp(self, node):
        """Handles unary operations (e.g., negation)."""
        operand = self.visit(node.operand)
        return self.operations[type(node.op)](operand)

    def visit_Name(self, node):
        """Handles variable names."""
        if node.id in self.context:
            return self.context[node.id]
        raise NameError(f"Name '{node.id}' is not defined in the given context.")

    def visit_Call(self, node):
        """Disallow function calls for security."""
        raise ValueError("Function calls are not allowed in formulas.")
