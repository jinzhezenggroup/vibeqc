"""Value-only auxiliary-g lowering of the common Gaussian moment algebra.

The f/f/g class needs F10, beyond the qualified five-root Rys domain. Emit
polynomial coefficients from the existing moment DAG and integrate with the
same Boys routine as DF response. Orbital g and g derivatives stay excluded.
"""

from itertools import product

from .cuda import CudaEmitter
from .df_derivatives import axis_polynomial
from .df_derivatives_cuda import (
    emit_df_boys_cuda,
    emit_df_geometry_cuda,
    emit_df_polynomial_dot_cuda,
)


def emit_df_g_values_cuda() -> str:
    """Emit bounded value helpers inside the generated DF value namespace.

    Only 85 axis cases are needed: f/f/g and g/s/g metric components. Each
    primitive owns at most three eleven-coefficient polynomials and F0..F10.
    No recurrence workspace grows with the molecular basis or batch size.
    """
    lines = [
        "namespace auxiliary_g {",
        "__device__ __forceinline__ double component(Vec3 a,unsigned i) { return i==0?a.x:i==1?a.y:a.z; }",
        "__device__ __forceinline__ unsigned power(Angular a,unsigned i) { return i==0?a.x:i==1?a.y:a.z; }",
        emit_df_boys_cuda(),
        "__device__ __noinline__ void axis_polynomial(unsigned a,unsigned b,unsigned c,",
        "    double pa,double pb,double dx,double sx,double sy,double ip,double iq,double* out) {",
        "  switch(a*25U+b*5U+c) {",
    ]
    for a, b, c in product(range(5), range(4), range(5)):
        if a == 4 and b != 0:
            continue
        graph, roots = axis_polynomial(a, b, c, auxiliary_g=True)
        emitter = CudaEmitter(graph, {})
        emitter.emit(roots)
        lines += [f"    case {a * 25 + b * 5 + c}U: {{", *emitter.lines]
        lines += [
            f"      out[{i}]={emitter.reference(root)};" for i, root in enumerate(roots)
        ]
        lines += ["      return;", "    }"]
    lines += [
        "  }",
        "  for(unsigned i=0;i<11;++i) out[i]=NAN;",
        "}",
        emit_df_polynomial_dot_cuda(),
        "struct Geometry { double pa[3],pb[3],dx[3],sx,sy,ip,iq,prefactor,f[11]; };",
        emit_df_geometry_cuda(moments="boys_values(total,rho*distance,g.f,work);"),
        r"""
/** Value only; callers validate shell roles before entering this bounded domain. */
__device__ __noinline__ double evaluate(double alpha,Vec3 A,Angular a,
    double beta,Vec3 B,Angular b,double gamma,Vec3 C,Angular c) {
  const unsigned total=a.x+a.y+a.z+b.x+b.y+b.z+c.x+c.y+c.z;
  Geometry g;
  prepare_geometry(alpha,A,beta,B,gamma,C,total,g);
  double base[3][11];
  unsigned degree[3];
  for(unsigned axis=0;axis<3;++axis) {
    const unsigned na=power(a,axis),nb=power(b,axis),nc=power(c,axis);
    degree[axis]=na+nb+nc;
    axis_polynomial(na,nb,nc,g.pa[axis],g.pb[axis],g.dx[axis],
                    g.sx,g.sy,g.ip,g.iq,base[axis]);
  }
  return g.prefactor*dot(degree[0],base[0],degree[1],base[1],degree[2],base[2],g.f);
}
} // namespace auxiliary_g
""",
    ]
    return "\n".join(lines)
