"""Versioned derivative policy, lowered into small compile-time class traits."""

import json
import re
from pathlib import Path

from .policy import SCHEDULES, DfDerivativeTrial

MANIFEST = Path(__file__).resolve().parents[1] / "production_df_derivatives.json"


def load_manifest(path=MANIFEST):
    """Reject unavailable math and unbound promotions before generating CUDA."""
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("architectures"), dict
    ):
        raise ValueError("unsupported DF derivative production manifest")
    for architecture, profile in payload["architectures"].items():
        if not re.fullmatch(r"sm_[1-9][0-9]+", architecture):
            raise ValueError("invalid target architecture")
        if type(profile.get("qualified")) is not bool:
            raise ValueError("explicit endpoint qualification status required")
        rows = profile["kernels"]
        seen = set()
        for row in rows:
            if not re.fullmatch(r"[0-3]{3}", row["class"]):
                raise ValueError("invalid derivative class")
            trial = DfDerivativeTrial(
                tuple(map(int, row["class"])),
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
                raise ValueError(
                    "both endpoints, sanitizer and gradient evidence required"
                )
    return payload


def emit_policy(path=MANIFEST):
    """Keep parsing and evidence handling out of the native response hot path."""
    manifest = load_manifest(path)
    entries = [
        (architecture, profile, row)
        for architecture, profile in manifest["architectures"].items()
        for row in profile["kernels"]
    ]
    lines = [
        "// Generated architecture-specific DF derivative policy.",
        "#pragma once",
        "namespace vibeqc::scf::generated_df_shell {",
        "struct ProductionChoice { unsigned variant; bool rys,available,qualified; };",
        f"inline constexpr bool production_policy_available={'true' if entries else 'false'};",
        "template<unsigned A,unsigned B,unsigned C> struct DfProductionPolicy {",
        "  static constexpr ProductionChoice select(unsigned architecture) {",
    ]
    for architecture, profile, row in entries:
        a, b, c = map(int, row["class"])
        variant = SCHEDULES.index(row["schedule"])
        rys = str(row["lowering"] == "rys").lower()
        qualified = str(profile["qualified"]).lower()
        lines.extend(
            [
                f"    if constexpr(A=={a} && B=={b} && C=={c})",
                f"      if(architecture=={int(architecture[3:])}) return {{{variant},{rys},true,{qualified}}};",
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
