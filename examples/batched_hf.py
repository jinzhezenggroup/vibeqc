import numpy as np

from qce import Calculator


systems = [
    [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
    [("He", (0.0, 0.0, 0.0))],
    [("H", (-1.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.0)), ("H", (1.0, 0.0, 0.0))],
]

calculator = Calculator(method="rhf", basis="sto-3g", device="cpu")
with calculator.prepare_batch(systems, charges=[0, 0, 1]) as batch:
    cold = batch.execute(strict=True)
    warm = batch.execute(strict=True)

    moved_h2 = np.array([[0.0, 0.0, -0.75], [0.0, 0.0, 0.75]])
    moved = batch.execute([moved_h2, None, None], strict=True)

print("cold energies:", cold.energies)
print("cold iterations:", [item.iterations for item in cold.items])
print("warm iterations:", [item.iterations for item in warm.items])
print("bucket ids:", [item.bucket_id for item in cold.items])
print("moved H2 energy:", moved.items[0].energy)
