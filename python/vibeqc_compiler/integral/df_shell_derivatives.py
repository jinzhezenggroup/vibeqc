"""Shell-shared lowering of the existing weighted DF derivative moment DAG.

All s/p/d/f classes use the scalar generated geometry, Boys values and axis
moments. The original seven non-SSS s/p classes remain a comparison subset.
Bounded schedules change component ownership and shell packing, not the DAG.
No derivative tensor is produced: the consumer accumulates six independent
center coordinates and recovers the auxiliary center by translation.
"""

import typing
from dataclasses import dataclass
from itertools import product

from .cuda import CudaEmitter
from .df_derivatives import axis_polynomial

PROTOTYPE_CLASSES = tuple(a for a in product(range(2), repeat=3) if any(a))
SHELL_CLASSES = tuple(product(range(4), repeat=3))


@dataclass(frozen=True)
class ShellSchedule:
    """One bounded ownership variant; the scientific cache is unchanged."""

    component_lanes: int
    triples_per_block: int
    shared_bytes: int


def shell_schedule(angular: typing.Any, variant: typing.Any) -> typing.Any:
    """Bound warp, packed-warp and compact-subgroup variants below 48 KiB.

    Reserve 1 KiB for compiler/runtime shared state. Compact groups use at
    least four lanes, so all three axis preparers remain independent. Larger
    component blocks cycle across the same lanes without increasing storage.
    """
    if variant not in (0, 1, 2):
        raise ValueError("unknown generated shell schedule")
    components = 1
    for l in angular:
        components *= (l + 1) * (l + 2) // 2
    _, axis_size = axis_cache_layout(angular)
    group_bytes = 8 * (components + 3 * axis_size + 25)
    lanes = 32 if variant != 2 else min(32, max(4, 1 << (components - 1).bit_length()))
    limit = 1 if variant == 0 else min(128 // lanes, (48 * 1024 - 1024) // group_bytes)
    groups = 1 << (limit.bit_length() - 1)
    return ShellSchedule(lanes, groups, groups * group_bytes + 1024)


def axis_cache_layout(angular: typing.Any) -> typing.Any:
    """Pack moment polynomials with only the first orbital center raised.

    Raising B follows exactly from raising A plus the center displacement
    times the base moment. The rectangular cache therefore needs no raised-B
    boundary, while indexing stays independent of Cartesian component tables.
    """
    offsets, size = {}, 0
    for powers in product(
        range(angular[0] + 2), range(angular[1] + 1), range(angular[2] + 1)
    ):
        offsets[powers] = size
        size += sum(powers) + 1
    return offsets, size


def shell_work_model(angular: typing.Any) -> typing.Any:
    """Describe fixed work in the emitted lowering before compiler optimization.

    Dynamic Boys iterations and sparsity depend on the primitive geometry and
    actual folded weights. The caller multiplies each active component's loop
    count by the number of executed primitive products; it must not count
    inactive components or infer hardware instructions from these values.
    """
    if tuple(angular) not in SHELL_CLASSES:
        raise ValueError("shell work model requires an s/p/d/f angular triple")
    offsets, size = axis_cache_layout(angular)

    def powers(degree: typing.Any) -> typing.Any:
        return [
            (degree - row, row - z, z)
            for row in range(degree + 1)
            for z in range(row + 1)
        ]

    loops = []
    for components in product(*(powers(degree) for degree in angular)):
        degrees = tuple(sum(axis) for axis in zip(*components))
        loops.append(
            sum(
                (degrees[axis] + 2)
                * (degrees[(axis + 1) % 3] + 1)
                * (degrees[(axis + 2) % 3] + 1)
                for axis in range(3)
            )
        )
    specialized = max(angular) <= 1
    return {
        "boys_maximum_order": sum(angular) + 1,
        "axis_polynomial_calls": 0 if specialized else 3 * len(offsets),
        "specialized_prepare_axis_calls": 3 if specialized else 0,
        "generic_prepare_entries": 0 if specialized else 3 * len(offsets),
        "cache_coefficient_values": 3 * size,
        "component_convolution_iterations": loops,
    }


def select_shell_classes(classes: typing.Any = None) -> typing.Any:
    """Return a validated canonical subset; ordering never depends on callers."""
    if classes is None:
        return SHELL_CLASSES
    requested = tuple(tuple(angular) for angular in classes)
    if (
        not requested
        or any(angular not in SHELL_CLASSES for angular in requested)
        or len(set(requested)) != len(requested)
    ):
        raise ValueError("expected unique supported s/p/d/f shell classes")
    return tuple(angular for angular in SHELL_CLASSES if angular in requested)


def emit_df_shell_derivatives_cuda(*, classes: typing.Any = None) -> typing.Any:
    """Emit class-specialized cache construction from the shared polynomial IR."""
    selected = select_shell_classes(classes)
    lines = [
        r"""// Generated shell-shared weighted DF derivatives.
#ifndef VIBEQC_GENERATED_DF_SHELL_DERIVATIVES_CUH
#define VIBEQC_GENERATED_DF_SHELL_DERIVATIVES_CUH
#include "generated_df_derivatives.cuh"
namespace vibeqc::scf::generated_df_shell {
namespace scalar = generated_df_derivatives;
template<unsigned A,unsigned B,unsigned C> struct Shell;
template<unsigned A,unsigned B,unsigned C,unsigned Variant> struct Schedule;
struct Contracted { double gradient[3][3]; };

/** CCA index of one normalized expansion's Cartesian component. */
__device__ __forceinline__ unsigned cartesian_index(unsigned l,const unsigned char* a) {
  const unsigned r=l-a[0];return r*(r+1)/2+a[2];
}
template<unsigned L>
__device__ __forceinline__ scalar::Angular angular(unsigned i) {
  for(unsigned r=0;r<=L;++r)
    if(i<(r+1)*(r+2)/2) {
      const unsigned z=i-r*(r+1)/2;
      return {L-r,r-z,z};
    }
  return {0,0,0};
}

template<unsigned A,unsigned B,unsigned C>
struct Moments {
  static constexpr bool shared_root_state=false;
  static constexpr unsigned na=(A+1)*(A+2)/2,nb=(B+1)*(B+2)/2,nc=(C+1)*(C+2)/2;
  static constexpr unsigned components=na*nb*nc;
  // Both supported public representations use CCA identity expansions for s/p.
  // Higher Cartesian classes are injective too, but retain the general path
  // until independently qualified alongside spherical d/f fallback semantics.
  static constexpr bool public_weight_identity=A<2 && B<2 && C<2;
  static constexpr unsigned rows=B+1,columns=C+1;
  static constexpr unsigned axis_size=(A+2)*rows*columns*(A+B+C+3)/2;
  static constexpr unsigned cache_coefficient_values=3*axis_size;
  static constexpr unsigned polynomial_calls=(A<2 && B<2 && C<2)?0:3*(A+2)*rows*columns;
  static constexpr unsigned specialized_axis_calls=(A<2 && B<2 && C<2)?3:0;
  /** Exact loop-body evaluations for one active component, before optimization.
   * Keep this alongside accumulate so diagnostic counts follow its lowering.
   */
  __device__ static unsigned convolution_work(unsigned item) {
    const auto a=angular<A>(item/nb/nc),b=angular<B>(item/nc%nb),c=angular<C>(item%nc);
    unsigned degree[3];
    for(unsigned axis=0;axis<3;++axis)
      degree[axis]=scalar::power(a,axis)+scalar::power(b,axis)+scalar::power(c,axis);
    unsigned work=0;
    for(unsigned axis=0;axis<3;++axis)
      work+=(degree[axis]+2)*(degree[(axis+1)%3]+1)*(degree[(axis+2)%3]+1);
    return work;
  }
  /** Translation recovers the auxiliary center from the two independent ones. */
  __device__ static Contracted finish(const double* independent) {
    Contracted result{};
    for(unsigned axis=0;axis<3;++axis) {
      result.gradient[0][axis]=independent[axis];
      result.gradient[1][axis]=independent[3+axis];
      result.gradient[2][axis]=-independent[axis]-independent[3+axis];
    }
    return result;
  }
  /** Sum the exact coefficient lengths of preceding rectangular entries. */
  __device__ __forceinline__ static unsigned offset(unsigned a,unsigned b,unsigned c) {
    return a*rows*columns*(a+rows+columns-1)/2
         + b*columns*(2*a+b+columns)/2 + c*(a+b+1)+c*(c-1)/2;
  }
  /** High-angular caches distribute unique moment polynomials over all lanes.
   * The same scalar-generated moment DAG supplies every coefficient; the
   * raised-B moments follow from raised-A and the unraised moment below.
   */
  __device__ static void prepare(const scalar::Geometry& g,double* cache,unsigned lane,unsigned lanes) {
    constexpr unsigned entries=(A+2)*(B+1)*(C+1);
    for(unsigned item=lane;item<3*entries;item+=lanes) {
      const unsigned axis=item/entries,index=item%entries;
      const unsigned a=index/rows/columns,b=index/columns%rows,c=index%columns;
      scalar::axis_polynomial(a,b,c,g.pa[axis],g.pb[axis],g.dx[axis],g.sx,g.sy,g.ip,g.iq,
                              cache+axis*axis_size+offset(a,b,c));
    }
  }
  /** Integrate the other two axes once for both differentiated centers.
   * For each coefficient i, H_i=sum_jk v_j w_k F_(i+j+k) is independent
   * of the differentiated center. Contract its two analytic derivatives
   * together, rather than repeating that polynomial product for A and B.
   * alpha/beta and external weights stay fixed under nuclear response.
   */
  __device__ __forceinline__ static void accumulate(unsigned item,double alpha,double beta,
      const scalar::Geometry& g,const double* cache,double weight,double* out) {
    const auto a=angular<A>(item/nb/nc),b=angular<B>(item/nc%nb),c=angular<C>(item%nc);
    unsigned degree[3],index[3];
    for(unsigned axis=0;axis<3;++axis) {
      const auto x=scalar::power(a,axis),y=scalar::power(b,axis),z=scalar::power(c,axis);
      degree[axis]=x+y+z;index[axis]=axis*axis_size+offset(x,y,z);
    }
    for(unsigned axis=0;axis<3;++axis) {
      const unsigned other=(axis+1)%3,last=(axis+2)%3;
      const auto x=scalar::power(a,axis),y=scalar::power(b,axis),z=scalar::power(c,axis);
      const double* v=cache+index[other];const double* w=cache+index[last];
      const double* raised_a=cache+axis*axis_size+offset(x+1,y,z);
      const double* base=cache+index[axis];
      const double displacement=g.pb[axis]-g.pa[axis];
      const double* lowered_a=x?cache+axis*axis_size+offset(x-1,y,z):raised_a;
      const double* lowered_b=y?cache+axis*axis_size+offset(x,y-1,z):raised_a;
      double value_a=0,value_b=0;
      for(unsigned i=0;i<=degree[axis]+1;++i) {
        double other_moment=0;
        for(unsigned j=0;j<=degree[other];++j)
          for(unsigned k=0;k<=degree[last];++k)
            other_moment+=v[j]*w[k]*g.f[i+j+k];
        value_a+=(2*alpha*raised_a[i]-(x && i<degree[axis]?x*lowered_a[i]:0.0))*other_moment;
        // (r-B)=(r-A)+(A-B); no second raised moment cache is needed.
        const double raised_b=raised_a[i]+displacement*(i<=degree[axis]?base[i]:0.0);
        value_b+=(2*beta*raised_b-(y && i<degree[axis]?y*lowered_b[i]:0.0))*other_moment;
      }
      out[axis]+=weight*g.prefactor*value_a;
      out[3+axis]+=weight*g.prefactor*value_b;
    }
  }
};
"""
    ]
    lines += [
        "template<unsigned A,unsigned B,unsigned C> struct Shell : Moments<A,B,C> {};"
    ]
    for angular in product(range(2), repeat=3):
        if angular not in selected:
            continue
        offsets, _ = axis_cache_layout(angular)
        parameters = ",".join(map(str, angular))
        lines += [
            f"template<> struct Shell<{parameters}> : Moments<{parameters}> {{",
            "  __device__ static void prepare_axis(const scalar::Geometry& g,unsigned axis,double* out) {",
            "    const double pa=g.pa[axis],pb=g.pb[axis],dx=g.dx[axis];",
            "    const double sx=g.sx,sy=g.sy,ip=g.ip,iq=g.iq;",
        ]
        for powers, offset in offsets.items():
            graph, roots = axis_polynomial(*powers)
            emitter = CudaEmitter(graph, {})
            emitter.emit(roots)
            lines += ["    {", *emitter.lines]
            lines += [
                f"      out[{offset + i}]={emitter.reference(root)};"
                for i, root in enumerate(roots)
            ]
            lines += ["    }"]
        lines += [
            "  }",
            "  __device__ static void prepare(const scalar::Geometry& g,double* cache,unsigned lane,unsigned) {",
            "    if(lane<3) prepare_axis(g,lane,cache+lane*axis_size);",
            "  }",
            "};",
        ]
    for angular in selected:
        for variant in range(3):
            schedule = shell_schedule(angular, variant)
            parameters = ",".join(map(str, (*angular, variant)))
            lines += [
                f"template<> struct Schedule<{parameters}> {{",
                f"  static constexpr unsigned lanes={schedule.component_lanes};",
                f"  static constexpr unsigned groups={schedule.triples_per_block};",
                f"  static constexpr unsigned shared_bytes={schedule.shared_bytes};",
                "};",
            ]
    if classes is None:
        lines += ["template<class Function> void for_each_class(Function function) {"]
        for angular in selected:
            lines += [
                f"  function.template operator()<{','.join(map(str, angular))}>();"
            ]
        lines += ["}"]
    lines += ["} // namespace vibeqc::scf::generated_df_shell", "#endif", ""]
    return "\n".join(lines)
