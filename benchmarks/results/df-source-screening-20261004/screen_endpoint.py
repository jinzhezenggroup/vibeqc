"""Reproduce the screening A/B through the unchanged acceptance comparator.

The comparator clears inherited DF controls at every endpoint. Apply the one
experimental control at that boundary, preserving its scientific gates, frozen
warm inputs, independent references and complete endpoint timing.
"""

import os

from benchmarks import compare_df_direct_endpoint as endpoint

select = endpoint.select_control


def select_screen(route: str) -> None:
    """Change only the explicitly requested whole-shell force error budget."""
    select(route)
    os.environ["GENERATIVEQC_DF_SHELL_SCREEN_ABS"] = os.environ.get(
        "HF_DF_SHELL_SCREEN_ABS", "off"
    )


if __name__ == "__main__":
    endpoint.select_control = select_screen
    endpoint.main()
