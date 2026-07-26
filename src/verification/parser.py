import re


def normalize_expression(expression: str):

    expression = expression.strip()

    replacements = {

        "×": "*",
        "x": "*",
        "X": "*",

        "÷": "/",

        "^": "**",

        "−": "-",
        "–": "-",

        "√": "sqrt",

        " ": ""
    }

    for old, new in replacements.items():
        expression = expression.replace(old, new)

    expression = re.sub(r"\s+", "", expression)

    return expression