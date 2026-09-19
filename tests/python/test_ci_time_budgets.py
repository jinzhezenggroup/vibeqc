"""Keep routine feedback bounded without truncating full qualification jobs."""

import re
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "filename, job, routine_minutes",
    [("ci.yml", "python", 20), ("cumetal-cuda.yml", "cuda-tests", 30)],
)
def test_full_qualification_has_a_separate_finite_budget(
    filename, job, routine_minutes
):
    path = Path(__file__).resolve().parents[2] / ".github/workflows" / filename
    section = path.read_text().split(f"\n  {job}:\n", 1)[1]
    match = re.search(
        r"timeout-minutes: \$\{\{ \(github.event_name == 'schedule' \|\| "
        r"github.event_name == 'workflow_dispatch'\) && (\d+) \|\| (\d+) \}\}",
        section,
    )
    assert match, "full qualification must not inherit the routine timeout"
    full, routine = map(int, match.groups())
    assert routine == routine_minutes
    assert 60 <= full <= 120
    if filename == "ci.yml":
        assert '!= "schedule"' in section and '!= "workflow_dispatch"' in section
