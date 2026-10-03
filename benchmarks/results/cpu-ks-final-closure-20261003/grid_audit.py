"""Independent same-coordinate weight construction; no native weights are inputs."""

def independent_same_coordinate_weights(topology, native_points, native_owners):
    import numpy as np
    from generativeqc_compiler.dft.grid import partition_weights
    # These raw factors depend solely on the fixed atoms and GridSpec: independent
    # NumPy Legendre nodes/weights, explicit radii and angular measure. Neither
    # native atomic nor molecular weights are accessible through this interface.
    raw = np.concatenate([tile.weights for tile in topology._raw_tiles(256)])
    owners = np.asarray(native_owners, dtype=np.int64)
    if native_points.shape != (topology.npoint, 3) or owners.shape != (topology.npoint,):
        raise RuntimeError('Same-coordinate input shape changed')
    expected = np.empty(topology.npoint, dtype=np.float64)
    for begin in range(0, topology.npoint, 256):
        end = min(begin + 256, topology.npoint)
        partition = partition_weights(native_points[begin:end], topology.centers,
            iterations=topology.spec.partition_iterations,
            coincident_tolerance=topology.spec.coincident_tolerance)
        expected[begin:end] = raw[begin:end] * partition[np.arange(end-begin), owners[begin:end]]
    return raw, expected
