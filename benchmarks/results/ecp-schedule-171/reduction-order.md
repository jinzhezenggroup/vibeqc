# Production radial accumulation order

The review on PR #458 correctly identifies a change in the internal kernel's
thread mapping. Its conclusion about competing radial threads does not apply
to the old production call site: the old kernel always receives one layer.

## Call-site evidence

Both the original measured base `b78215e` and integrated master `1865c5a`
contain this host loop and contraction invocation:

```cpp
for (unsigned r = 0; r < nr; ++r) {
  // AO evaluation and projection use the same stream and one layer.
  contract<<<(size + 127) / 128, 128, 0, stream>>>(
      daos, dterms, system.ecp_terms.size(), dsphere, dradii + r,
      values, projections, n, nq, 1, c, ncoord, destination);
}
```

The literal `1` is the kernel's `nr` argument. All these launches use the
same stream, so layer r completes before the next layer's contraction.
The old `i >= nr * n * n` guard consequently restricts execution to `n*n`
threads, with decoded radial index zero. There is exactly one active thread
per unique AO pair after the `b > a` guard.

Sources (immutable full files, including the kernel signature and callers):

- [Measured base b78215e](https://github.com/jinzhezenggroup/vibeqc/blob/b78215ef0bf2cb43c91234c9c2dc852151605468/src/integrals/ecp_cuda.cu)
- [Integrated master 1865c5a](https://github.com/jinzhezenggroup/vibeqc/blob/1865c5a/src/integrals/ecp_cuda.cu)
- [Validated candidate 0eb9384](https://github.com/zhaiwenxi/vibeqc/blob/0eb938490f0bf1588d168831d26518c870f8aacf/src/integrals/ecp_cuda.cu)

The baseline Nsight trace retained here records 161 contraction launches for
161 radial layers, independently corroborating the one-layer launch schedule.
The candidate records 41 launches and handles the last one-layer tail.

## Scope of the preserved contract

The candidate changes the internal mapping to one thread per unique pair and
a loop over the current batch. Ordered batches on the same stream and the
ascending inner loop retain the production radial update sequence. It retains
each FP64 atomic addition instead of combining several layers into a tile sum.
Local/nonlocal outputs and transpose entries remain separate addresses;
coincident A/B/C centers receive their updates in the same thread order.

No claim is made that the old kernel would have had the same ordering if a
different caller supplied multiple layers. Nor is this a claim that complete
energies or forces are bitwise reproducible across compilers or devices. The
independent raw-matrix/derivative, endpoint and sanitizer gates in the original
and merged evidence remain the acceptance criteria.

The clarification at `f0ee7ec` changed documentation only. All 28 measured
source files at that commit matched `merge-source-identity.json`, the passing
CI at `0eb9384` and the retained GPU run. The later allocator error-origin
correction is documented separately in `merge-validation.md`.
