// Included in the native test namespace. CPU AO values independently determine
// each expected map, then full-global-AO bilinears validate the selected E/V.
void ao_discovery_cases() {
  for (bool spherical : {false, true}) {
    auto molecule = system(3, spherical);
    molecule.shells.push_back({0, 3, {{0.51, 1.0}}});
    molecule.shells.push_back({0, 3, {{0.39, 1.0}}});
    molecule.shells.push_back({1, 3, {{0.32, 1.0}}});
    molecule.shells.push_back({1, 2, {{0.27, 1.0}}});
    const AoBasis basis(molecule);
    const MolecularGrid grid(molecule, {1, 7, 7, 8, 3, 1e-12});
    require(grid.point_count() > 257 && basis.nao > 32 && basis.nao % 32,
            "discovery oracle must cross both native block boundaries");
    std::vector<double> ao(4 * grid.point_count() * basis.nao);
    basis.evaluate(grid.points().data(), grid.point_count(), 1, 0, basis.nao, ao.data(), ao.size());
    for (std::size_t tile : {129U, 257U})
      for (double cutoff : {1e-16, 1e-2, 1e8}) {
        CudaXcAoTiles expected;
        expected.offsets.push_back(0);
        std::uint64_t point_square = 0;
        std::size_t empty = 0;
        for (std::size_t first = 0; first < grid.point_count(); first += tile) {
          const auto count = std::min(tile, grid.point_count() - first);
          for (std::size_t mu = 0; mu < basis.nao; ++mu) {
            double maximum = 0;
            for (std::size_t point = first; point < first + count; ++point)
              for (unsigned jet = 0; jet < 4; ++jet)
                maximum = std::max(
                    maximum, std::abs(ao[(jet * grid.point_count() + point) * basis.nao + mu]));
            if (maximum > cutoff) expected.indices.push_back(mu);
          }
          const auto active = expected.indices.size() - expected.offsets.back();
          point_square += count * active * active;
          empty += active == 0;
          expected.offsets.push_back(expected.indices.size());
        }
        Fixture fixture(basis, grid, 4U, spherical, tile, CudaXcAoPrecision::Fp64, false, 1, 1,
                        nullptr, true);
        const auto bounds = cuda_xc_ao_selection_resources(fixture.layout);
        require(!fixture.plan->select_local_ao(cutoff, bounds.host_peak_bytes - 1),
                "AO discovery exceeded its host admission");
        require(!fixture.plan->layout().local_ao, "budget miss published a partial map");
        require(fixture.plan->select_local_ao(cutoff, bounds.host_peak_bytes),
                "exact AO discovery admission failed");
        const auto& work = fixture.plan->ao_selection_work();
        require(work.selected && work.active_sum == expected.indices.size() &&
                    work.empty_tiles == empty && work.point_ao_square_sum == point_square &&
                    work.discovery_ao_jet_values == ao.size(),
                "AO discovery differs from independent CPU masks or work counts");
        local_ao_reference(fixture, basis, grid, expected, density(basis.nao, spherical ? 2 : 1));
        bool rejected = false;
        try {
          fixture.plan->select_local_ao(cutoff, bounds.host_peak_bytes);
        } catch (const std::invalid_argument&) {
          rejected = true;
        }
        require(rejected, "AO discovery accepted a previously evaluated/local plan");
        fixture.canary();
      }
    Fixture dense(basis, grid, 4U, false, 129);
    require(!dense.plan->select_local_ao(1e-16, std::numeric_limits<std::size_t>::max()),
            "AO discovery exceeded its device admission");
    local_ao_reference(dense, basis, grid, local_maps(grid.point_count(), 129, basis.nao, 0),
                       density(basis.nao, 1));
    // A physical replay body touches scratch before its generation is
    // published. Zero submitted/evaluation counters must not admit discovery
    // into a previously captured or outstanding body.
    Fixture replay(basis, grid, 4U, false, 129, CudaXcAoPrecision::Fp64, false, 1, 1, nullptr,
                   true);
    const auto d = density(basis.nao, 1);
    check(cudaMemcpyAsync(replay.density, d.data(), d.size() * sizeof(double),
                          cudaMemcpyHostToDevice, replay.stream));
    replay.plan->enqueue_replay_body(replay.density, d.size());
    require(replay.plan->transfers().evaluations == 0, "replay probe unexpectedly published work");
    bool rejected = false;
    try {
      replay.plan->select_local_ao(1e-16, std::numeric_limits<std::size_t>::max());
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "discovery was allowed after an unpublished replay body");
    check(cudaStreamSynchronize(replay.stream));
    replay.canary();
  }
}
