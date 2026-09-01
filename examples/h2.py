from vibeqc import Calculator

calculator = Calculator(method="rhf", basis="sto-3g", device="cuda")
result = calculator.singlepoint(
    [
        ("H", (0.0, 0.0, -0.7)),
        ("H", (0.0, 0.0, 0.7)),
    ]
)

print(f"energy = {result.energy:.12f} Eh")
print(f"forces =\n{result.forces}")
print(f"scientific backend = {result.executed_backend}")
