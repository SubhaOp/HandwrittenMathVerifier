import sympy as sp


def evaluate(expression):

    try:

        value = sp.simplify(expression)

        return value

    except Exception:

        return None