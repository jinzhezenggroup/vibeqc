"""Coverage findings must follow source changes rather than pinned answers."""

from pathlib import Path

import pytest

from tools import audit_libxc_maple_coverage as coverage


@pytest.mark.parametrize("conditional", (False, True))
def test_missing_functional_entry_is_not_directly_importable(
    tmp_path: Path, conditional: bool
) -> None:
    source = "(* type: lda_exc *)\ng := (rs, z) -> rs:\n"
    if conditional:
        source += "$ifdef ENABLE_ENTRY\nf := (rs, z) -> g(rs,z):\n$endif\n"
    (tmp_path / "trial.mpl").write_text(source)
    item = coverage.audit(tmp_path)["sources"][0]
    assert item["parse_failures"] == 0
    assert item["direct_importable"] is False
    assert "missing callable entry f" in item["entry_blockers"]


@pytest.mark.parametrize("functional", (False, True))
def test_markdown_reports_actual_failures(tmp_path: Path, functional: bool) -> None:
    prefix = "(* type: lda_exc *)\n" if functional else ""
    (tmp_path / "broken.mpl").write_text(prefix + "f := (rs,z) -> rs ! 2:\n")
    text = coverage.render_markdown(coverage.audit(tmp_path))
    assert (
        "No functional source has a parser-syntax failure" not in text or not functional
    )
    assert "util.mpl is intentionally" not in text
    assert "lda_x.mpl reaches" not in text
    assert "mgga_x_rscan.mpl reaches" not in text


def test_clean_report_does_not_invent_evaluator_gaps(tmp_path: Path) -> None:
    (tmp_path / "simple.mpl").write_text("(* type: lda_exc *)\nf := (rs,z) -> rs:\n")
    report = coverage.audit(tmp_path)
    assert report["direct_importable_functionals"] == 1
    report["importer_semantics"] = "future-importer-version"
    text = coverage.render_markdown(report)
    assert "Current v10 gaps" not in text
    assert "lda_x.mpl reaches" not in text
    assert "mgga_x_rscan.mpl reaches" not in text
