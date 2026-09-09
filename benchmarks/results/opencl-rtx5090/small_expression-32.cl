// Scientific identity: 8cd61e12f2b4305d1af004fa6588246be74f21c9b37bfe059efd7178204ca256
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void small_expression(__global const double* inputs,
    __global double* outputs, ulong count) {
  const size_t item = get_global_id(0);
  if (item >= count) return;
  const double input_column_0 = inputs[item * 2 + 0];
  const double input_column_1 = inputs[item * 2 + 1];
  const double v0 = input_column_0 * 2;
  const double v1 = input_column_1 + v0;
  outputs[item * 1 + 0] = v1;
}
