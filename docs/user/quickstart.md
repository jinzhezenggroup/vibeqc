# Quick start

The Python API uses Bohr for coordinates, Hartree for energies, and Hartree/Bohr for forces.

```python
from vibeqc import Calculator

calc = Calculator(method="rhf", basis="sto-3g", device="cuda")
result = calc.singlepoint([
    ("H", (0.0, 0.0, -0.7)),
    ("H", (0.0, 0.0,  0.7)),
])

print(result.energy)
print(result.forces)
```

Discover current public methods with:

```bash
vibeqc methods
vibeqc methods --json
```

The generated [public method table](../public_methods.md) is the documentation-side source of truth for canonical method identities and declared capabilities.
