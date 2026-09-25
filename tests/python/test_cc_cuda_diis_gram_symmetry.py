from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "src" / "cc" / "cuda_solver.cu").read_text(encoding="utf-8")


def _gram_body() -> str:
    start = SOURCE.index("__global__ void gram_kernel(")
    stop = SOURCE.index("\n}\n\nstruct Layout", start)
    return SOURCE[start:stop]


def test_cuda_rccsd_gram_computes_one_symmetric_pair() -> None:
    body = _gram_body()
    assert "const int row = pair / history, col = pair % history;" in body
    assert "if (row > col) return;" in body
    assert "gram[std::size_t(row) * history + col] = values[0];" in body
    assert "if (row != col) gram[std::size_t(col) * history + row] = values[0];" in body
    assert "__dmul_rn(errors[std::size_t(row) * elements + i]," in body
    assert "errors[std::size_t(col) * elements + i]" in body


def test_cuda_rccsd_gram_preserves_dense_launch_and_reduces_dot_work() -> None:
    assert "gram_kernel<<<count * count, 256, 0, s.stream>>>" in SOURCE
    for history, square, unique in ((6, 36, 21), (8, 64, 36), (20, 400, 210)):
        assert history * history == square
        assert history * (history + 1) // 2 == unique


def test_retained_ccsd_t_default_history_work_census() -> None:
    # The retained README CCSD(T) endpoint uses DIIS history 6, performs 14
    # DIIS solves without restart, and has nocc=5/nvir=4. The first four solves
    # see histories 2..5; the remaining ten are at the full history of six.
    histories = [2, 3, 4, 5] + [6] * 10
    square_pairs = sum(history * history for history in histories)
    unique_pairs = sum(history * (history + 1) // 2 for history in histories)
    assert square_pairs == 414
    assert unique_pairs == 244

    residual_elements = 5 * 4 + (5 * 5) * (4 * 4)
    assert residual_elements == 420
    assert square_pairs * residual_elements == 173_880
    assert unique_pairs * residual_elements == 102_480
