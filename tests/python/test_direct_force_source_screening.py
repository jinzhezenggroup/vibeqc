"""Exercise requested-source force screening on the actual native scalar helper.

The host harness binds four synthetic shells; real-device molecular force and
independent ERI checks separately qualify the changed radial consumer.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_force_screening_honors_requested_sources(tmp_path: Path) -> None:
    """Keep live spin channels and exact thresholds; exclude inactive J/K work."""
    HEADER = ROOT / "src/scf/cuda/direct_screening.cuh"
    text = HEADER.read_text()
    start = text.index("template <bool Unrestricted, DirectScreeningPurpose Purpose>")
    end = text.index("/** Safely reject a complete shell-pair-block", start)
    function = (
        text[start:end]
        .replace("__device__ ", "")
        .replace("__forceinline__ ", "inline ")
    )
    source = (
        r"""
    #include <array>
    #include <cmath>
    #include <cstddef>
    #include <cstdint>
    #include <cstdio>
    #include <cstdlib>
    #include <limits>
    using std::fmax;
    using std::fmin;
    enum class DirectScreeningPurpose { Fock, Force };
    constexpr double kForceDensityProductScreeningTolerance = 1e-14;
    struct ShellPairDensityBounds { double coulomb, exchange_alpha, exchange_beta; };
    struct DeviceBatch {
      const std::int32_t* shell_pair_systems;
      const std::int32_t* shell_pair_first;
      const std::int32_t* shell_pair_second;
    };
    std::size_t system_shell_pair_index(const DeviceBatch&, int, int i, int j) {
      if (i < j) { int k=i; i=j; j=k; }
      return static_cast<std::size_t>(i*(i+1)/2+j);
    }
    """
        + function
        + r"""
    unsigned comparisons=0;
    void require(bool condition, const char* label) {
      ++comparisons;
      if (!condition) { std::fprintf(stderr,"failed screen: %s\n",label); std::exit(1); }
    }
    template<bool U>
    void cases() {
      std::int32_t systems[10]{};
      const std::int32_t first[10]{0,1,1,2,2,2,3,3,3,3};
      const std::int32_t second[10]{0,0,1,0,1,2,0,1,2,3};
      DeviceBatch batch{systems,first,second};
      std::array<double,10> schwarz; schwarz.fill(1.0);
      std::array<ShellPairDensityBounds,10> bounds{};
      const auto accepts = [&](bool exchange, bool coulomb, double tolerance=1e-12) {
        return direct_shell_quartet_survives_screening<U,DirectScreeningPurpose::Force>(
          batch,1,8,tolerance,schwarz.data(),bounds.data(),nullptr,exchange,coulomb);
      };
      // (10|32): the Coulomb shell blocks are occupied, while every cross-block
      // exchange density is zero. A K-only derivative has no live contraction.
      bounds[1].coulomb=bounds[8].coulomb=1.0;
      require(accepts(false,false),"mixed preserves live Coulomb");
      require(accepts(false,true),"J-only preserves live Coulomb");
      require(!accepts(true,false),"K-only removes Coulomb-only task");
      // Cross densities survive the first (linear) screen but their product lies
      // below the stricter force gate; large inactive J must not retain the task.
      bounds[7].exchange_alpha=bounds[3].exchange_alpha=1e-8;
      require(!accepts(true,false),"K-only uses its force product gate");
      // A live exchange product must survive independently of Coulomb screening.
      bounds[7].exchange_alpha=bounds[3].exchange_alpha=1.0;
      bounds[1].coulomb=bounds[8].coulomb=0.0;
      require(accepts(true,false),"alpha exchange live");
      require(accepts(false,false),"mixed exchange live");
      require(!accepts(false,true),"J-only removes exchange-only task");
      // In UHF the beta channel is an independent source, not a cancellation with
      // alpha. Restricted exchange must not read the unused beta field.
      bounds[7].exchange_alpha=bounds[3].exchange_alpha=0.0;
      bounds[7].exchange_beta=bounds[3].exchange_beta=1.0;
      require(accepts(true,false)==U,"beta-only spin contract");
      // Check equality and neighbors at min(requested,1e-14), with unit factors
      // avoiding ambiguous product rounding in the comparison itself.
      bounds.fill({0.0,0.0,0.0});
      for (double tol : {1e-9,1e-12,1e-16}) {
        const double gate=std::fmin(tol,1e-14);
        bounds[7].exchange_alpha=1.0;
        bounds[3].exchange_alpha=gate;
        require(accepts(true,false,tol),"at force gate");
        bounds[3].exchange_alpha=std::nextafter(gate,0.0);
        require(!accepts(true,false,tol),"below force gate");
        bounds[3].exchange_alpha=std::nextafter(gate,1.0);
        require(accepts(true,false,tol),"above force gate");
      }
      // J and K remain separately admitted even if a physical weighted sum might
      // cancel; the mask never uses a signed combined density coefficient.
      bounds.fill({1.0,1.0,1.0});
      require(accepts(false,false),"no cross-source cancellation");
      require(accepts(true,false),"K source independent");
      require(accepts(false,true),"J source independent");
      require(!accepts(true,true),"empty source mask");
    }
    int main() { cases<false>(); cases<true>(); std::printf("%u source-demand comparisons\n",comparisons); }
    """
    )
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("host screening execution requires c++ and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    probe = tmp_path / "screening-probe"
    probe.with_suffix(".cpp").write_text(source)
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            str(probe.with_suffix(".cpp")),
            "-o",
            str(probe),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    result = subprocess.run(
        [str(probe)], check=True, capture_output=True, text=True, timeout=10
    )
    assert result.stdout.strip() == "42 source-demand comparisons"
