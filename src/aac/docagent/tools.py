"""The calculator tool. Expressions are parsed with `ast` and evaluated node by node: no eval(),
no names, no attribute access, and limits on size so "9**9**9" cannot hang the process."""

from __future__ import annotations

import ast
import math
import operator
import re

from langchain_core.tools import tool

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"round": round, "min": min, "max": max, "abs": abs, "sqrt": math.sqrt}
MAX_LEN = 200
MAX_EXPONENT = 100
MAX_MAGNITUDE = 1e15


class CalculatorError(ValueError):
    pass


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError("exponent too large")
        try:
            value = _BINARY[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise CalculatorError("division by zero") from exc
        if abs(value) > MAX_MAGNITUDE:
            raise CalculatorError("result too large")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise CalculatorError(f"unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> str:
    expr = expression.strip().replace("^", "**").replace("×", "*")
    expr = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expr)  # 43,200 -> 43200, but round(x, 2) keeps its comma
    expr = expr.rstrip("=").strip()
    if not expr or len(expr) > MAX_LEN:
        raise CalculatorError(f"expression must be 1-{MAX_LEN} characters")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"not a valid arithmetic expression: {expression!r}") from exc
    value = _eval(tree.body)
    if isinstance(value, float):
        value = round(value, 6)
        if value.is_integer():
            value = int(value)
    return str(value)


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as "18 * 12 * 12 * 0.8".
    Supports + - * / // % ** and round(), min(), max(), abs(), sqrt(). Use it for any calculation;
    do not use it to look up facts."""
    try:
        return calculate(expression)
    except CalculatorError as exc:
        return f"error: {exc}"


TOOLS = [calculator]
