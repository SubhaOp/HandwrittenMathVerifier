from src.verification.verifier import MathVerifier
from src.verification.utils import display

verifier = MathVerifier()

tests = [

    "2+3=5",

    "2+3=6",

    "12*8=96",

    "12*8=90",

    "10/2=5",

    "(2+5)*3=21",

    "2^5=32",

    "sqrt(49)=7"

]

for equation in tests:

    result = verifier.verify(equation)

    display(result)