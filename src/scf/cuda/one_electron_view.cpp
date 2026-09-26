#include "scf/cuda/one_electron_view.hpp"

namespace vibeqc::scf::cuda_execution {

OneElectronDeviceView one_electron_view(const DeviceBatch& batch) {
  return {batch.batch_size,
          batch.nbf,
          static_cast<std::size_t>(batch.total_shell_pairs),
          batch.atom_offsets,
          batch.atomic_numbers,
          batch.positions,
          batch.shell_atoms,
          batch.shell_ao_offsets,
          batch.shell_primitive_offsets,
          batch.shell_pair_first,
          batch.shell_pair_second,
          batch.ao_shells,
          batch.ao_term_counts,
          batch.ao_term_angular,
          batch.ao_term_coefficients,
          batch.primitive_exponents,
          batch.primitive_coefficients};
}

}  // namespace vibeqc::scf::cuda_execution
