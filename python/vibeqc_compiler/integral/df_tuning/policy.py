"""Typed derivative trials and fail-closed, profile-local class selection."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import typing
from dataclasses import asdict, dataclass
from itertools import product

from ..df_rys_shell import RYS_SHELL_CLASSES
from ..df_shell_derivatives import shell_schedule

LOW_ANGULAR_CLASSES = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 0, 2),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (2, 0, 0),
)
SCHEDULES = ("warp", "packed", "compact")
# Availability is explicit: adding another lowering extends this registry;
# unsupported requests never fall back to another mathematical algorithm.
LOWERINGS = {
    "polynomial": frozenset(product(range(4), repeat=3)),
    "rys": frozenset(RYS_SHELL_CLASSES),
}


@dataclass(frozen=True)
class DfDerivativeTrial:
    """One mathematical lowering and ownership schedule for an angular class."""

    angular: tuple[int, int, int]
    lowering: str
    variant: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.angular, tuple)
            or len(self.angular) != 3
            or any(type(x) is not int or x not in range(4) for x in self.angular)
        ):
            raise ValueError("expected an s/p/d/f angular triple")
        if (
            self.lowering not in LOWERINGS
            or self.angular not in LOWERINGS[self.lowering]
        ):
            raise ValueError("unsupported DF derivative lowering/class")
        if type(self.variant) is not int or self.variant not in range(len(SCHEDULES)):
            raise ValueError("unsupported DF derivative schedule")

    @property
    def class_name(self) -> typing.Any:
        return "".join(map(str, self.angular))

    @property
    def key(self) -> typing.Any:
        return f"{self.class_name}:{self.lowering}:{SCHEDULES[self.variant]}"

    @property
    def symbol(self) -> typing.Any:
        return "df_" + self.key.replace(":", "_")

    @property
    def block_threads(self) -> typing.Any:
        schedule = shell_schedule(self.angular, self.variant)
        return schedule.component_lanes * schedule.triples_per_block

    def artifact_key(
        self,
        *,
        generator_sha256: typing.Any,
        architecture: typing.Any,
        toolchain: typing.Any,
    ) -> typing.Any:
        """Bind a trial to generated/runtime source, target and actual compiler."""
        if not all(
            isinstance(x, str) and x
            for x in (generator_sha256, architecture, toolchain)
        ):
            raise ValueError("complete generator/target/toolchain identity required")
        payload = {
            "schema": 1,
            "trial": asdict(self),
            "generator": generator_sha256,
            "architecture": architecture,
            "toolchain": toolchain,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def enumerate_trials(angular: typing.Any = LOW_ANGULAR_CLASSES) -> typing.Any:
    """Enumerate every available lowering/schedule without invented Rys entries."""
    return tuple(
        DfDerivativeTrial(tuple(a), lowering, variant)
        for a in angular
        for lowering, available in LOWERINGS.items()
        if tuple(a) in available
        for variant in range(len(SCHEDULES))
    )


def read_profile(
    payload: typing.Any, *, angular: typing.Any = LOW_ANGULAR_CLASSES
) -> typing.Any:
    """Read the #398 ledger, preserving exact signature frequencies and work.

    A signature may have fewer active than visited tasks. Both counts are kept;
    primitive/component counts are measured rather than inferred from angular
    degree or dense symmetry assumptions. Duplicate rows and absent counts are
    errors because they could otherwise silently bias a weighted class score.
    """
    rows = {}
    for cls in payload["classes"]:
        shell_class = tuple(cls["angular"])
        if shell_class not in angular:
            continue
        for signature in cls["signatures"]:
            primitives = tuple(signature["primitives"])
            if len(primitives) != 3 or any(
                type(x) is not int or x < 1 for x in primitives
            ):
                raise ValueError("invalid primitive signature")
            key = (shell_class, primitives)
            if key in rows:
                raise ValueError("duplicate workload signature")
            work = {
                k: signature["work"][k]
                for k in (
                    "shell_tasks",
                    "active_shell_tasks",
                    "primitive_products",
                    "active_component_products",
                )
            }
            if any(type(x) is not int or x < 0 for x in work.values()):
                raise ValueError("invalid measured work")
            if (
                not 0 < work["shell_tasks"]
                or work["active_shell_tasks"] > work["shell_tasks"]
            ):
                raise ValueError("invalid active shell domain")
            if work["primitive_products"] != work["active_shell_tasks"] * math.prod(
                primitives
            ):
                raise ValueError("primitive work disagrees with its signature")
            rows[key] = work
    if not rows:
        raise ValueError("profile contains no supported derivative work")
    return rows


def rank_profiles(
    profiles: typing.Any,
    results: typing.Any,
    *,
    baselines: typing.Any,
    minimum_gain: typing.Any = 0.03,
    angular: typing.Any = LOW_ANGULAR_CLASSES,
) -> typing.Any:
    """Select one trial per class only when all retained profiles agree.

    Each result covers one complete real signature workload. Timings are raw
    repeated CUDA-event samples; counts must equal the baseline's executed
    domain. A missing signature, nonfinite datum, numerical failure or rejected
    compile invalidates the entire class candidate. The profile sums retain
    workload frequencies; profiles are never averaged together. Results remain
    diagnostic until full-gradient, sanitizer and endpoint gates are attached.
    """
    if not math.isfinite(minimum_gain) or not 0 <= minimum_gain < 1:
        raise ValueError("minimum_gain must be in [0,1)")
    indexed = {}
    for row in results:
        key = (row["profile"], row["candidate"], tuple(row["primitives"]))
        if key in indexed:
            raise ValueError("duplicate candidate/signature result")
        indexed[key] = row
    rankings = {}
    for name, payload in profiles.items():
        domain = read_profile(payload, angular=angular)
        class_rows = {}
        for shell_class in sorted({a for a, _ in domain}):
            cls = "".join(map(str, shell_class))
            baseline = baselines[cls]
            scores = {}
            rejections = {}
            for trial in enumerate_trials((shell_class,)):
                samples_total = 0.0
                reasons = []
                for a, primitives in domain:
                    if a != shell_class:
                        continue
                    row = indexed.get((name, trial.key, primitives))
                    reference = indexed.get((name, baseline, primitives))
                    if row is None or reference is None:
                        reasons.append("incomplete signature coverage")
                        continue
                    if (
                        row.get("eligible") is not True
                        or row.get("numerical_passed") is not True
                    ):
                        reasons.append("compile/resource/numerical gate failed")
                    if (
                        row.get("work") != reference.get("work")
                        or row.get("work") != domain[(a, primitives)]
                    ):
                        reasons.append("work domain differs")
                    samples = row.get("milliseconds", [])
                    if len(samples) < 5 or any(
                        type(x) not in (float, int) or not math.isfinite(x) or x <= 0
                        for x in samples
                    ):
                        reasons.append("invalid or insufficient timing samples")
                    else:
                        samples_total += statistics.median(samples)
                if reasons:
                    rejections[trial.key] = sorted(set(reasons))
                else:
                    scores[trial.key] = samples_total
            winner = baseline
            if baseline in scores:
                best = min(scores, key=lambda key: (scores[key], key))
                if scores[best] < scores[baseline] * (1 - minimum_gain):
                    winner = best
            class_rows[cls] = {
                "baseline": baseline,
                "winner": winner,
                "milliseconds": scores,
                "rejections": rejections,
                "baseline_valid": baseline in scores,
            }
        rankings[name] = class_rows
    combined, conflicts = {}, {}
    classes = sorted({c for rows in rankings.values() for c in rows})
    for cls in classes:
        choices = {
            name: rows[cls]["winner"] for name, rows in rankings.items() if cls in rows
        }
        valid = len(choices) == len(profiles) and all(
            rows[cls]["baseline_valid"] for rows in rankings.values() if cls in rows
        )
        if valid and len(set(choices.values())) == 1:
            combined[cls] = next(iter(choices.values()))
        else:
            conflicts[cls] = choices
    return {
        "profiles": rankings,
        "proposed_mapping": combined,
        "conflicts": conflicts,
        "production_promoted": False,
        "minimum_gain": minimum_gain,
    }
