"""Pinned Libxc Maple importer coverage audit."""

from tools import audit_libxc_maple_coverage as coverage


def test_pinned_libxc_maple_inventory_and_functional_parser_coverage() -> None:
    report = coverage.audit()
    assert report["importer_semantics"] == "libxc-maple-graph/v11"
    assert report["source_files"] == 27
    assert report["functional_files"] == 23
    assert report["support_files"] == 4

    functional = [item for item in report["sources"] if item["kind"] == "functional"]
    assert all(item["parse_failures"] == 0 for item in functional)


def test_static_direct_import_blockers_are_explicit() -> None:
    report = coverage.audit()
    blocked = {
        item["file"]: item["entry_blockers"]
        for item in report["sources"]
        if item["kind"] == "functional" and not item["direct_importable"]
    }
    assert report["direct_importable_functionals"] == 21
    assert blocked == {
        "lda_x.mpl": ["lda_x_spin"],
        "mgga_x_rscan.mpl": ["mgga_exchange_nsp"],
    }


def test_checked_in_coverage_snapshot_is_current() -> None:
    report = coverage.audit()
    expected = coverage.render_markdown(report)
    actual = (coverage.ROOT / "docs/libxc_maple_coverage.md").read_text(
        encoding="utf-8"
    )
    assert actual == expected


def test_default_audit_uses_canonical_vendored_sources() -> None:
    """The normal audit must not depend on the retired external source tree."""
    assert coverage.LIBXC_ROOT == coverage.ROOT / "upstream/libxc/7.0.0"
    assert (coverage.LIBXC_ROOT / "lda_x.mpl").is_file()
