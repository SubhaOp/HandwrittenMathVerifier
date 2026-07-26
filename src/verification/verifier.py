from src.verification.parser import normalize_expression
from src.verification.evaluator import evaluate


class MathVerifier:

    def verify(self, expression):

        expression = normalize_expression(expression)

        if "=" not in expression:

            return {

                "status": "Invalid",

                "message": "No '=' found."
            }

        left, right = expression.split("=")

        left_value = evaluate(left)

        right_value = evaluate(right)

        if left_value is None or right_value is None:

            return {

                "status": "Error",

                "message": "Unable to evaluate."
            }

        if left_value == right_value:

            return {

                "status": "Correct",

                "expression": expression,

                "answer": str(left_value)
            }

        return {

            "status": "Incorrect",

            "expression": expression,

            "predicted_answer": str(right_value),

            "correct_answer": str(left_value)
        }