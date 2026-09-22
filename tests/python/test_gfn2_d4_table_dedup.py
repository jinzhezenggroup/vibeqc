from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_gfn2_runtime_reuses_canonical_d4_tables() -> None:
    runtime = ROOT / "src/xtb/gfn2_runtime"
    d4_source = runtime / "src/model/gfn2/d4.cpp"

    assert not (runtime / "data/parameters/d4.hpp").exists()
    assert not (runtime / "data/parameters/d4_c6_part0.inc").exists()
    assert not (runtime / "data/parameters/d4_c6_part1.inc").exists()

    source = d4_source.read_text(encoding="utf-8")
    assert '#include "dft/dispersion/d4_data.hpp"' in source
    assert '#include "dft/dispersion/d4_reference.hpp"' in source
    assert "shared::evaluate_d4_fixed_charge" in source
    assert "d4_data::kElements" in source
    assert "d4_data::kReferences" in source
    assert "d4_data::kReferenceC6" in source
    assert "high * (high + 1u) / 2u + low" in source
