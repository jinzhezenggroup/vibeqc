// Included inside the native CUDA detail namespace after its primitive helpers.
// Keep the validated handwritten force schedules available for same-binary gates.

__global__ void one_electron_force_scalar_kernel(DeviceBatch batch, const std::int32_t* pair_first,
                                                 const std::int32_t* pair_second,
                                                 std::size_t pair_count, const double* density,
                                                 const double* weighted_density,
                                                 const std::uint8_t* active, double* forces) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch.batch_size) * pair_count) {
    return;
  }
  const std::int32_t system = static_cast<std::int32_t>(element / pair_count);
  const std::size_t local = element % pair_count;
  if (active[system] == 0) return;
  const std::size_t i = static_cast<std::size_t>(pair_first[local]);
  const std::size_t j = static_cast<std::size_t>(pair_second[local]);
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * matrix_size;
  const double pij = density[matrix_offset + matrix_index(i, j, n)];
  const double wij = weighted_density[matrix_offset + matrix_index(i, j, n)];
  if (pij == 0.0 && wij == 0.0) return;
  const std::int64_t ao_i =
      static_cast<std::int64_t>(system) * batch.nbf + static_cast<std::int32_t>(i);
  const std::int64_t ao_j =
      static_cast<std::int64_t>(system) * batch.nbf + static_cast<std::int32_t>(j);
  const unsigned maximum =
      batch.shell_angular[batch.ao_shells[ao_i]] + batch.shell_angular[batch.ao_shells[ao_j]] + 1U;
#define VIBEQC_ONE_ELECTRON_FORCE_CASE(Order)                                                  \
  case Order:                                                                                  \
    contracted_one_electron_force_pair<Order>(batch, system, static_cast<std::int32_t>(i),     \
                                              static_cast<std::int32_t>(j), pij, wij, forces); \
    break
  switch (maximum) {
    VIBEQC_ONE_ELECTRON_FORCE_CASE(1);
    VIBEQC_ONE_ELECTRON_FORCE_CASE(2);
    VIBEQC_ONE_ELECTRON_FORCE_CASE(3);
    VIBEQC_ONE_ELECTRON_FORCE_CASE(4);
    VIBEQC_ONE_ELECTRON_FORCE_CASE(5);
    VIBEQC_ONE_ELECTRON_FORCE_CASE(6);
    VIBEQC_ONE_ELECTRON_FORCE_CASE(7);
  }
#undef VIBEQC_ONE_ELECTRON_FORCE_CASE
}

/**
 * Treat nuclei as a prepared point-charge auxiliary dimension.
 *
 * One warp owns one AO pair and reuses a shared primitive-pair Hermite table
 * while its lanes evaluate independent nuclear centers. This avoids a host or
 * device launch per nucleus and preserves the public-basis density contraction
 * used by the scalar accuracy oracle.
 */
__global__ void one_electron_force_cooperative_kernel(DeviceBatch batch,
                                                      const std::int32_t* pair_first,
                                                      const std::int32_t* pair_second,
                                                      std::size_t pair_count, const double* density,
                                                      const double* weighted_density,
                                                      const std::uint8_t* active, double* forces) {
  extern __shared__ double one_electron_shared[];
  if (blockDim.x != warpSize) return;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x);
  if (element >= static_cast<std::size_t>(batch.batch_size) * pair_count) {
    return;
  }
  const std::int32_t system = static_cast<std::int32_t>(element / pair_count);
  const std::size_t local = element % pair_count;
  if (active[system] == 0) return;
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t i = static_cast<std::size_t>(pair_first[local]);
  const std::size_t j = static_cast<std::size_t>(pair_second[local]);
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * matrix_size;
  const double pij = density[matrix_offset + matrix_index(i, j, n)];
  const double wij = weighted_density[matrix_offset + matrix_index(i, j, n)];
  if (pij == 0.0 && wij == 0.0) return;
  const std::int64_t ao_i =
      static_cast<std::int64_t>(system) * batch.nbf + static_cast<std::int32_t>(i);
  const std::int64_t ao_j =
      static_cast<std::int64_t>(system) * batch.nbf + static_cast<std::int32_t>(j);
  const unsigned maximum =
      batch.shell_angular[batch.ao_shells[ao_i]] + batch.shell_angular[batch.ao_shells[ao_j]] + 1U;
  auto* shared_coefficients =
      reinterpret_cast<OneElectronDerivativeHermiteCoefficients*>(one_electron_shared);
#define VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE(Order)                                    \
  case Order:                                                                                \
    contracted_one_electron_force_pair_cooperative<Order>(                                   \
        batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j), pij, wij, \
        forces, shared_coefficients);                                                        \
    break
  switch (maximum) {
    VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE(1);
    VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE(2);
    VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE(3);
    VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE(4);
    VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE(5);
    VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE(6);
    VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE(7);
  }
#undef VIBEQC_ONE_ELECTRON_FORCE_COOPERATIVE_CASE
}
