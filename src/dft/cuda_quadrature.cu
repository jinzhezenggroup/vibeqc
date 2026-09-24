// CUDA allocation/execution owner. Mathematics, tiling and exact device layout
// come from the scientific compiler; the scalar reference remains in grid.cpp.
#include <algorithm>
#include <stdexcept>

#include "dft/grid.hpp"
#include "generated_quadrature.cuh"
#include "runtime/cuda_resources.cuh"

namespace vibeqc::dft {
namespace q = generated::quadrature;

std::size_t cuda_quadrature_bytes(std::size_t atoms, std::size_t points) {
  return q::layout(atoms, points).device_bytes;
}

MolecularGrid MolecularGrid::from_cuda(const core::System& system, GridSpec spec, int device) {
  MolecularGrid result(system, spec, Deferred{});
  const auto per_atom =
      q::product(q::product(spec.radial_points, spec.angular_polar), spec.angular_azimuth);
  const auto l = q::layout(system.atoms.size(), q::product(system.atoms.size(), per_atom));
  const auto [rn, rw] = legendre_rule(spec.radial_points);
  const auto [zn, zw] = legendre_rule(spec.angular_polar);
  // These small shared quadrature tables are input rules, not a CPU molecular
  // grid. The generated kernels own coordinates and every partition weight.
  std::vector<double> input(l.polar, 0.0);
  for (std::size_t a = 0; a < l.atoms; ++a) {
    std::copy(system.atoms[a].position.begin(), system.atoms[a].position.end(),
              input.begin() + 3 * a);
    const double radius = spec.element_radii[system.atoms[a].atomic_number];
    input[3 * l.atoms + a] = radius > 0.0 ? radius : 1.0;
  }
  std::copy(rn.begin(), rn.end(), input.begin() + l.rules);
  std::copy(rw.begin(), rw.end(), input.begin() + l.rules + 512);
  std::copy(zn.begin(), zn.end(), input.begin() + l.rules + 1024);
  std::copy(zw.begin(), zw.end(), input.begin() + l.rules + 1280);
  result.points_.resize(q::product(3, l.points));
  result.weights_.resize(l.points);
  result.owners_.resize(l.points);
  for (std::size_t a = 0; a < l.atoms; ++a)
    std::fill_n(result.owners_.begin() + a * per_atom, per_atom, static_cast<std::uint32_t>(a));

  runtime::CudaDeviceScope guard(device);
  runtime::OwnedCudaStream stream(device);
  runtime::OwnedCudaBuffer<double> storage(device, l.doubles, stream.get());
  runtime::OwnedCudaBuffer<int> invalid(device, 1, stream.get());
  const auto check = runtime::cuda_resource_check;
  double* data = storage.get();
  check(cudaMemcpyAsync(data, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice,
                        stream.get()));
  check(cudaMemsetAsync(invalid.get(), 0, sizeof(int), stream.get()));
  q::polar_kernel<<<q::blocks(spec.angular_polar), 128, 0, stream.get()>>>(
      spec.angular_polar, data + l.rules + 1024, data + l.rules + 1280, data + l.polar);
  check(cudaGetLastError());
  q::azimuth_kernel<<<q::blocks(spec.angular_azimuth), 128, 0, stream.get()>>>(spec.angular_azimuth,
                                                                               data + l.azimuth);
  check(cudaGetLastError());
  q::geometry_kernel<<<q::blocks(l.atoms * l.atoms), 128, 0, stream.get()>>>(data, l.atoms,
                                                                             data + l.geometry);
  check(cudaGetLastError());
  for (std::size_t begin = 0; begin < l.points; begin += l.tile) {
    const auto count = std::min(l.tile, l.points - begin);
    q::points_kernel<<<q::blocks(count), 128, 0, stream.get()>>>(
        begin, count, spec.radial_points, spec.angular_polar, spec.angular_azimuth, data,
        data + 3 * l.atoms, data + l.rules, data + l.rules + 512, data + l.polar, data + l.azimuth,
        data + l.xyz, data + l.weights);
    check(cudaGetLastError());
    q::distances_kernel<<<q::atom_point_grid(count, l.atoms), 128, 0, stream.get()>>>(
        count, l.atoms, data + l.xyz, data, data + l.distances);
    check(cudaGetLastError());
    q::launch_partition(spec.partition_iterations, count, l.atoms, spec.coincident_tolerance,
                        data + l.distances, data + l.geometry, data + l.logs, stream.get());
    check(cudaGetLastError());
    q::normalize_kernel<<<q::blocks(count), 128, 0, stream.get()>>>(
        begin, count, per_atom, l.atoms, data + l.logs, data + l.weights, invalid.get());
    check(cudaGetLastError());
    // Current grid/derivative APIs own host arrays. Include this staging in
    // complete preparation time; scratch is retired before XC plan creation.
    check(cudaMemcpyAsync(result.points_.data() + 3 * begin, data + l.xyz,
                          3 * count * sizeof(double), cudaMemcpyDeviceToHost, stream.get()));
    check(cudaMemcpyAsync(result.weights_.data() + begin, data + l.weights, count * sizeof(double),
                          cudaMemcpyDeviceToHost, stream.get()));
  }
  int bad = 0;
  check(cudaMemcpyAsync(&bad, invalid.get(), sizeof(int), cudaMemcpyDeviceToHost, stream.get()));
  stream.synchronize();
  if (bad) throw std::runtime_error("invalid CUDA Becke partition normalization");
  return result;
}
}  // namespace vibeqc::dft
