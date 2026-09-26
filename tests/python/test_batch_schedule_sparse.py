"""Feasible zero-size graphs must not allocate a dense degree census."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("operation", ["scatter", "gather"])
def test_huge_empty_domains_have_bounded_diagnostics(operation: str) -> None:
    pytest.importorskip("resource")
    if not Path("/proc/self/statm").exists():
        pytest.skip("address-space-bounded regression requires Linux procfs")
    program = r"""
import os, resource, sys
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import Index, IndexSpace, Program, TensorSpec, input_tensor, indexed_gather, scatter_add
from vibeqc_compiler.tensor.cuda_plan import plan_cuda
huge = 2**40
zero = Index("empty", IndexSpace("empty", "component", 0))
source_size, target_size = (0, huge) if sys.argv[1] == "scatter" else (huge, 0)
src = Index("source", IndexSpace("source", "pair", source_size))
dst = Index("target", IndexSpace("target", "atom", target_size))
x = input_tensor("x", TensorSpec((zero, src), role="input"))
op = scatter_add if sys.argv[1] == "scatter" else indexed_gather
plan = plan_cuda(Program({"out": op(x, 1, (), dst)}), cuda_target_info("sm_80"))
assert plan.index_table_bytes == 0
# Deny the original multi-terabyte allocation in this child process only.
# Imports and plan construction are complete before imposing the ceiling.
pages = int(open("/proc/self/statm").read().split()[0])
soft = pages * os.sysconf("SC_PAGE_SIZE") + (64 << 20)
_, hard = resource.getrlimit(resource.RLIMIT_AS)
if hard != resource.RLIM_INFINITY:
    soft = min(soft, hard)
resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
row = plan.batch_schedule.ragged_steps[0]
assert row.edge_count == row.nonempty_groups == row.max_degree == 0
assert row.empty_groups == huge
assert row.degree_histogram == ((0, huge),)
assert row.scan_work == row.scheduled_work == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", program, operation],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("operation", ["scatter", "gather"])
def test_sparse_histograms_match_dense_small_domains(operation: str) -> None:
    from collections import Counter
    from itertools import product

    from vibeqc_compiler.common.cuda_target import cuda_target_info
    from vibeqc_compiler.tensor import (
        Index,
        IndexSpace,
        Program,
        TensorSpec,
        indexed_gather,
        input_tensor,
        scatter_add,
    )
    from vibeqc_compiler.tensor.cuda_plan import plan_cuda

    outer = Index("outer", IndexSpace("components", "component", 2))
    target = cuda_target_info("sm_80")
    for groups in range(5):
        for edges in range(5):
            for positions in product(range(groups), repeat=edges):
                nsource, ntarget = (
                    (edges, groups) if operation == "scatter" else (groups, edges)
                )
                source = Index("src", IndexSpace("source", "pair", nsource))
                dest = Index("dst", IndexSpace("target", "atom", ntarget))
                value = input_tensor("x", TensorSpec((outer, source), role="input"))
                op = scatter_add if operation == "scatter" else indexed_gather
                program = Program({"out": op(value, 1, positions, dest)})
                row = plan_cuda(program, target).batch_schedule.ragged_steps[0]
                dense = [0] * groups
                for position in positions:
                    dense[position] += 1
                assert row.degree_histogram == tuple(sorted(Counter(dense).items()))
                assert row.empty_groups == dense.count(0)
                assert row.nonempty_groups == groups - dense.count(0)
                assert row.max_degree == max(dense, default=0)
                assert row.edge_count == edges
                assert row.scheduled_work == 2 * edges
