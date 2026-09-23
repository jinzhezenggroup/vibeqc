from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_import_probe(source: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_extension_package_root_keeps_public_surfaces_lazy() -> None:
    _run_import_probe(
        """
        import sys

        import vibeqc.extensions as extensions

        assert extensions.API_VERSION == 1
        assert "vibeqc.extensions.method" not in sys.modules
        assert "vibeqc.extensions.tensor" not in sys.modules
        assert "vibeqc.extensions.xc" not in sys.modules
        assert {"method", "tensor", "xc"}.issubset(dir(extensions))
        """
    )


def test_selecting_one_extension_surface_does_not_activate_siblings() -> None:
    _run_import_probe(
        """
        import sys

        from vibeqc.extensions import method

        assert method.__name__ == "vibeqc.extensions.method"
        assert "vibeqc.extensions.method" in sys.modules
        assert "vibeqc.extensions.tensor" not in sys.modules
        assert "vibeqc.extensions.xc" not in sys.modules
        """
    )
