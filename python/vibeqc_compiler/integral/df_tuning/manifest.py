"""Versioned derivative policy, lowered into small compile-time class traits."""

import json
import re
import typing
from pathlib import Path

from .policy import SCHEDULES, DfDerivativeTrial

MANIFEST = Path(__file__).resolve().parents[1] / "production_df_derivatives.json"


def _validate_profile(profile: typing.Any) -> None:
    """Apply the same mathematical and evidence gates to both comparison arms."""
    if not isinstance(profile, dict) or type(profile.get("qualified")) is not bool:
        raise ValueError("explicit endpoint qualification status required")
    rows = profile["kernels"]
    seen = set()
    for row in rows:
        if not re.fullmatch(r"[0-3]{3}", row["class"]):
            raise ValueError("invalid derivative class")
        angular = tuple(map(int, row["class"]))
        trial = DfDerivativeTrial(
            (angular[0], angular[1], angular[2]),
            row["lowering"],
            SCHEDULES.index(row["schedule"]),
        )
        if trial.angular in seen:
            raise ValueError("duplicate production derivative class")
        seen.add(trial.angular)
    if rows:
        evidence = profile.get("provenance", {})
        if not all(
            isinstance(evidence.get(k), str) and evidence[k]
            for k in (
                "generator_sha256",
                "toolchain",
                "profile_sha256",
                "candidate_report",
            )
        ):
            raise ValueError("complete candidate provenance required")
        if profile["qualified"] and not all(
            isinstance(evidence.get(k), str) and evidence[k]
            for k in (
                "endpoint_384",
                "endpoint_768",
                "sanitizer",
                "gradient_fixtures",
            )
        ):
            raise ValueError("both endpoints, sanitizer and gradient evidence required")


def load_manifest(path: typing.Any = MANIFEST) -> typing.Any:
    """Reject unavailable math and unbound promotions before generating CUDA.

    An unqualified campaign may retain one qualified ``baseline`` profile per
    architecture. Automatic execution keeps that baseline; the existing
    ``candidate`` selector admits the proposed mapping. This allows interleaved
    endpoints with one prepared state and one arena, without per-class controls
    or two simultaneously resident multi-gigabyte libraries/DF plans.
    """
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("architectures"), dict
    ):
        raise ValueError("unsupported DF derivative production manifest")
    for architecture, profile in payload["architectures"].items():
        if not re.fullmatch(r"sm_[1-9][0-9]+", architecture):
            raise ValueError("invalid target architecture")
        _validate_profile(profile)
        if "baseline" in profile:
            baseline = profile["baseline"]
            _validate_profile(baseline)
            if (
                profile["qualified"]
                or not baseline["qualified"]
                or "baseline" in baseline
            ):
                raise ValueError(
                    "a campaign requires one qualified baseline and an unqualified candidate"
                )
    return payload


def emit_policy(path: typing.Any = MANIFEST) -> typing.Any:
    """Keep parsing and evidence handling out of the native response hot path."""
    manifest = load_manifest(path)
    entries = []
    for architecture, profile in manifest["architectures"].items():
        arms = [(profile, "true")]
        if "baseline" in profile:
            arms = [(profile, "candidate"), (profile["baseline"], "!candidate")]
        for arm, selected in arms:
            entries.extend((architecture, arm, row, selected) for row in arm["kernels"])
    lines = [
        "// Generated architecture-specific DF derivative policy.",
        "#pragma once",
        "namespace vibeqc::scf::generated_df_shell {",
        "struct ProductionChoice { unsigned variant; bool rys,available,qualified; };",
        f"inline constexpr bool production_policy_available={'true' if entries else 'false'};",
        "template<unsigned A,unsigned B,unsigned C> struct DfProductionPolicy {",
        "  static constexpr ProductionChoice select(unsigned architecture, [[maybe_unused]] bool candidate=false) {",
    ]
    for architecture, profile, row, selected in entries:
        a, b, c = map(int, row["class"])
        variant = SCHEDULES.index(row["schedule"])
        rys = str(row["lowering"] == "rys").lower()
        qualified = str(profile["qualified"]).lower()
        lines.extend(
            [
                f"    if constexpr(A=={a} && B=={b} && C=={c})",
                f"      if(architecture=={int(architecture[3:])} && {selected}) return {{{variant},{rys},true,{qualified}}};",
            ]
        )
    lines.extend(
        [
            "    return {3,false,false,false};",
            "  }",
            "};",
            "} // namespace vibeqc::scf::generated_df_shell",
            "",
        ]
    )
    return "\n".join(lines)
