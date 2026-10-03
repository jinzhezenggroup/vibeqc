// Included inside test_dft_cuda.cu's test namespace. These independent full-AO
// CPU bilinears zero omitted columns point by point; they do not use the GPU's
// compact density or triangular potential contraction schedule.

CudaXcAoTiles local_maps(std::size_t points, std::size_t tile_points, std::size_t nao,
                         unsigned variant) {
  CudaXcAoTiles maps;
  maps.offsets.push_back(0);
  for (std::size_t begin = 0; begin < points; begin += tile_points) {
    const auto tile = begin / tile_points;
    for (std::size_t ao = 0; ao < nao; ++ao)
      if (variant == 0 || (variant == 1 && tile % 4 != 0 && (ao + tile) % 3 != 0))
        maps.indices.push_back(ao);
    maps.offsets.push_back(maps.indices.size());
  }
  return maps;
}

void local_ao_reference(Fixture& fixture, const AoBasis& basis, const MolecularGrid& grid,
                        const CudaXcAoTiles& maps, const std::vector<double>& d) {
  const auto count = grid.point_count(), n = basis.nao;
  const auto captured = fixture.submit_density_features(d);
  const auto result = fixture.scalars();
  require(result.error == 0, "local AO XC reported a device error");
  std::vector<double> ao(4 * count * n), expected(d.size(), 0.0);
  basis.evaluate(grid.points().data(), count, 1, 0, n, ao.data(), ao.size());
  double energy = 0.0, electrons[2]{};
  for (std::size_t p = 0; p < count; ++p) {
    const auto tile = p / fixture.layout.tile_points;
    const auto phi = [&](unsigned jet, std::size_t mu) {
      if (!std::binary_search(maps.indices.begin() + maps.offsets[tile],
                              maps.indices.begin() + maps.offsets[tile + 1], mu))
        return 0.0;
      return ao[(jet * count + p) * n + mu];
    };
    double rho[2]{}, gradient[2][3]{}, tau[2]{};
    for (unsigned spin = 0; spin < 2; ++spin) {
      const bool restricted = fixture.layout.spins == 1;
      const auto channel = restricted ? 0U : spin;
      const double scale = restricted ? 0.5 : 1.0;
      for (std::size_t mu = 0; mu < n; ++mu)
        for (std::size_t nu = 0; nu < n; ++nu) {
          const auto value = scale * d[(channel * n + mu) * n + nu];
          rho[spin] += phi(0, mu) * value * phi(0, nu);
          for (unsigned k = 0; k < 3; ++k) {
            gradient[spin][k] +=
                value * (phi(k + 1, mu) * phi(0, nu) + phi(0, mu) * phi(k + 1, nu));
            tau[spin] += 0.5 * value * phi(k + 1, mu) * phi(k + 1, nu);
          }
        }
      electrons[spin] += grid.weights()[p] * rho[spin];
    }
    close(captured.first[p], rho[0] + rho[1], "local AO captured density", 2e-12);
    for (unsigned k = 0; k < 3; ++k)
      close(captured.second[3 * p + k], gradient[0][k] + gradient[1][k],
            "local AO captured gradient", 2e-12);
    const auto xc = evaluate_wb97mv_point(rho, gradient, tau);
    require(std::isfinite(xc.energy), "local AO CPU point reference rejected valid features");
    energy += grid.weights()[p] * xc.energy;
    for (unsigned spin = 0; spin < fixture.layout.spins; ++spin)
      for (std::size_t mu = 0; mu < n; ++mu)
        for (std::size_t nu = 0; nu < n; ++nu) {
          double value = xc.rho[spin] * phi(0, mu) * phi(0, nu);
          for (unsigned k = 0; k < 3; ++k) {
            value +=
                xc.gradient[spin][k] * (phi(k + 1, mu) * phi(0, nu) + phi(0, mu) * phi(k + 1, nu));
            value += xc.kinetic[spin] * phi(k + 1, mu) * phi(k + 1, nu);
          }
          expected[(spin * n + mu) * n + nu] += grid.weights()[p] * value;
        }
  }
  close(result.energy, energy, "local AO CPU/GPU energy", 2e-11);
  for (unsigned spin = 0; spin < 2; ++spin)
    close(result.electrons[spin], electrons[spin], "local AO CPU/GPU electron count", 2e-11);
  const auto actual = fixture.potential();
  for (std::size_t i = 0; i < expected.size(); ++i)
    close(actual[i], expected[i], "local AO CPU/GPU potential",
          2e-11 + 2e-12 * std::abs(expected[i]));
  fixture.canary();
}

void local_ao_cases() {
  for (bool spherical : {false, true}) {
    auto molecule = system(3, spherical);
    molecule.shells.push_back({0, 3, {{0.51, 1.0}}});
    molecule.shells.push_back({0, 3, {{0.39, 1.0}}});
    molecule.shells.push_back({1, 3, {{0.32, 1.0}}});
    molecule.shells.push_back({1, 2, {{0.27, 1.0}}});
    const AoBasis basis(molecule);
    const MolecularGrid grid(molecule, {1, 3, 3, 4, 3, 1e-12});
    require(basis.nao > 32, "local AO fixture must span scalar and tiled schedules");
    for (bool uks : {false, true})
      for (std::size_t tile : {7U, 19U})
        for (unsigned variant : {0U, 1U, 2U}) {
          auto maps = local_maps(grid.point_count(), tile, basis.nao, variant);
          const auto reference_maps = maps;
          Fixture fixture(basis, grid, 4U, uks, tile, CudaXcAoPrecision::Fp64, false, 1.0, 1.0,
                          &maps);
          const auto dense = cuda_xc_layout(basis, grid, 4U, uks, tile);
          require(fixture.layout.device_bytes ==
                      dense.device_bytes + maps.indices.size() * sizeof(std::size_t),
                  "local AO indices were not charged exactly");
          require(fixture.layout.host_ao_map_bytes == maps.offsets.size() * sizeof(std::size_t),
                  "local AO host offsets were not charged exactly");
          // Setup copies all indices and offsets. Releasing caller metadata must
          // not leave any borrowed host storage in asynchronous evaluation.
          maps = {};
          auto d = density(basis.nao, uks ? 2U : 1U);
          local_ao_reference(fixture, basis, grid, reference_maps, d);
          for (auto& value : d) value *= 0.73;
          local_ao_reference(fixture, basis, grid, reference_maps, d);
          nonlocal_potential_case(basis, grid, uks, &reference_maps, tile);
          if (tile == 19) graph_capture(basis, grid, 4U, uks, tile, &reference_maps);
        }
    const auto dense = cuda_xc_layout(basis, grid, 4U, false, 19);
    const auto valid = local_maps(grid.point_count(), 19, basis.nao, 0);
    for (unsigned failure = 0; failure < 5; ++failure) {
      auto bad = valid;
      if (failure == 0) bad.offsets.pop_back();
      if (failure == 1) bad.indices[1] = bad.indices[0];
      if (failure == 2) bad.indices[0] = basis.nao;
      if (failure == 3) bad.offsets[1] = bad.indices.size() + 1;
      if (failure == 4) std::swap(bad.indices[0], bad.indices[1]);
      bool rejected = false;
      try {
        (void)cuda_xc_local_ao_layout(dense, bad);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      require(rejected, "malformed local AO map was accepted");
    }
  }
}
