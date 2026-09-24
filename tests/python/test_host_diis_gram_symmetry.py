from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/solver/diis.hpp"


def _update_body() -> str:
    source = SOURCE.read_text()
    start = source.index("  std::vector<double> update(")
    end = source.index("\n private:", start)
    return source[start:end]


def test_host_diis_gram_computes_each_symmetric_pair_once() -> None:
    body = _update_body()
    assert "for (std::size_t j = i; j < n; ++j)" in body
    assert body.count("std::inner_product(") == 1
    assert "gram[i * n + j] = dot;" in body
    assert "gram[j * n + i] = dot;" in body
    assert "scale = std::max(scale, std::abs(dot));" in body


def test_host_diis_gram_pair_work_census() -> None:
    expected = {
        2: (4, 3),
        6: (36, 21),
        8: (64, 36),
        20: (400, 210),
    }
    for history, (baseline, triangular) in expected.items():
        assert history * history == baseline
        assert history * (history + 1) // 2 == triangular
        assert triangular < baseline
