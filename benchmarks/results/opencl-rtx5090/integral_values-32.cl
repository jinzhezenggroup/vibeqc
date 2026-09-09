// Scientific identity: 343f553a4adefbdeb2de1a0bffc3d488eec9d7e15b0bd09cda4af72d1b6e8597
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__attribute__((reqd_work_group_size(32, 1, 1)))
__kernel void integral_values(__global const double* inputs,
    __global double* outputs, ulong count) {
  const size_t item = get_global_id(0);
  if (item >= count) return;
  const double input_column_0 = inputs[item * 9 + 0];
  const double input_column_1 = inputs[item * 9 + 1];
  const double input_column_2 = inputs[item * 9 + 2];
  const double input_column_3 = inputs[item * 9 + 3];
  const double input_column_4 = inputs[item * 9 + 4];
  const double input_column_5 = inputs[item * 9 + 5];
  const double input_column_6 = inputs[item * 9 + 6];
  const double input_column_7 = inputs[item * 9 + 7];
  const double input_column_8 = inputs[item * 9 + 8];
  const double v0 = input_column_7 * -2;
  const double v1 = v0 * v0;
  const double v2 = input_column_2 * input_column_2;
  const double v3 = v1 * v2;
  const double v4 = input_column_1 * v3;
  const double v5 = input_column_0 * v0;
  const double v6 = v4 + v5;
  const double v7 = input_column_5 * input_column_6;
  const double v8 = v7 * -1.0;
  const double v9 = v6 * v8;
  const double v10 = v9 * input_column_8;
  const double v11 = v10 * 34.986836655249725;
  const double v12 = input_column_3 * input_column_4;
  const double v13 = input_column_3 + input_column_4;
  const double v14 = sqrt(v13);
  const double v15 = v12 * v14;
  const double v16 = 1.0 / v15;
  const double v17 = v11 * v16;
  outputs[item * 1 + 0] = v17;
}
