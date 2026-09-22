# Energy, forces, gradients, and Hessians

For a fixed electronic method, energy depends on nuclear coordinates.

The **gradient** is the derivative of energy with respect to nuclear coordinates. The **force** has the opposite sign:

`force = - gradient`

A geometry optimizer uses first derivatives to search for stationary structures.

The **Hessian** contains second derivatives of the energy and describes local curvature. It is used for vibrational analysis, stationary-point characterization, and curvature-aware optimization. A Hessian-vector product applies this curvature without necessarily forming the full matrix.

Do not infer derivative support merely because an energy endpoint exists; support must be documented for the chosen method/backend.

Next: [post-HF methods](post-hf.md).
