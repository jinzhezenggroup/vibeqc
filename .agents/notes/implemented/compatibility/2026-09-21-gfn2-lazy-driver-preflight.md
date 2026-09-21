# Decision: preserve CPU preflight in a CUDA-enabled native library

Status: implemented repair in the GFN2 CUDA bootstrap
Date: 2026-09-21

## Problem

The CUDA bootstrap introduced a direct libcuda.so.1 dependency for two allocation
metadata queries. CUDA compilation succeeded, but loading the resulting library
for driver-free CPU preflight failed before any backend could be selected.
Ordinary CPU tests also attempted CUDA and matched incomplete diagnostic strings.

## Decision

On supported Linux ELF targets, reuse the repository's existing reviewed Implib
source generator for exactly cuGetErrorString and cuMemGetAddressRange_v2. Resolve
the real driver SONAME only when those queries are invoked. Keep direct SDK
linkage on other native platforms. Enable C/assembly languages in project scope;
do not hide language initialization inside a function. Vendor numerical and
allocation-validation bodies remain unchanged.

The optional molecular CUDA tests require VIBEQC_TEST_GFN2_CUDA=1 inside an
allocated Slurm job. Once explicitly requested, a missing runtime or numerical
failure fails the test instead of being converted into a skip.

## Evidence and boundaries

An actual CMake/C++/C/assembly linkage regression checks that the generated library
has no driver DT_NEEDED entry and its CPU entry loads without invoking the driver.
A separate fake driver validates the exact versioned query signatures and data.
This is a host ABI/linkage test, not a real GFN2 GPU energy/force qualification.
Full native CUDA build, actual molecular force comparisons and the stated resource
and replay gates remain necessary before promoting the bootstrap capability.

Agent: ChatGPT
Model: GPT-6 Astra Pro
