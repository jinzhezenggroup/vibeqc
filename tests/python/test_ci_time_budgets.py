"""Keep fast PR feedback bounded without truncating full qualification jobs."""

import re
from pathlib import Path


def test_full_python_qualification_has_a_separate_finite_budget():
    text = (
        Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"
    ).read_text()
    python_job = text.split("\n  python:\n", 1)[1].split("\n  upload-coverage:", 1)[0]
    match = re.search(
        r"timeout-minutes: \$\{\{ \(github.event_name == 'schedule' \|\| "
        r"github.event_name == 'workflow_dispatch'\) && (\d+) \|\| (\d+) \}\}",
        python_job,
    )
    assert match, "schedule/manual full suite must not inherit the routine PR timeout"
    full, routine = map(int, match.groups())
    assert routine == 20
    assert 60 <= full <= 120
    assert '!= "schedule"' in python_job and '!= "workflow_dispatch"' in python_job
