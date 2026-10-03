"""Fail-closed ownership of exactly supplied explicit quadrature, without rebuilds."""
import hashlib

def fingerprint(value):
    if value is None:
        raise RuntimeError('Explicit grid was reset to None')
    return dict(shape=list(value.shape),dtype=str(value.dtype),raw_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())

class ExplicitGridGuard:
    def __init__(self,grid,points,weights,*,expected_points=331776):
        self.grid=grid;self.points=points;self.weights=weights;self.build_attempts=0
        if points.shape!=(expected_points,3) or weights.shape!=(expected_points,):
            raise RuntimeError('Explicit grid shape differs from fixed point inventory')
        self.expected=dict(coords=fingerprint(points),weights=fingerprint(weights))
        # Policy attributes reset PySCF grids. Configure them BEFORE installation.
        grid.radii_adjust=None
        grid.coords=points;grid.weights=weights
        # The explicit inventory must never silently fall back to another grid.
        grid.build=self.deny_build
        self.check('after_binding')

    def deny_build(self,*args,**kwargs):
        self.build_attempts+=1
        raise RuntimeError('Forbidden rebuild of frozen explicit grid')

    def check(self,label):
        actual=dict(coords=fingerprint(self.grid.coords),weights=fingerprint(self.grid.weights))
        if self.grid.coords is not self.points or self.grid.weights is not self.weights:
            raise RuntimeError('Explicit grid array identity changed: '+label)
        if actual!=self.expected:
            raise RuntimeError('Explicit grid contents changed: '+label)
        if self.build_attempts:
            raise RuntimeError('Explicit grid rebuild was attempted: '+label)
        return dict(label=label,**actual,object_identity_preserved=True,build_attempts=0)
