"""Compatibility facade for separately owned tuning policy and execution.

Candidate analysis, CUDA emission, manifest provenance, resource gates, process
execution and CLI parsing live under tuning/. New callers should import their
semantic owner; these callable re-exports preserve existing entry points.
"""

from .tuning.analysis import (
    StaticAlgebraModel,
    _analysis_roots,
    _balanced_component_labels,
    _cached_static_algebra_model,
    _integral_signature,
    _packed_force_geometry_analysis,
    static_algebra_model,
)
from .tuning.cli import argument_parser, main
from .tuning.driver import _run_autotune
from .tuning.emission import (
    _class_name,
    _identifier_suffix,
    _isolate_schedule_symbols,
    _oracle_schedule_trial,
    _oracle_symbol_prefix,
    emit_schedule_driver,
    emit_schedule_oracle_translation_unit,
    emit_schedule_resource_translation_unit,
    emit_schedule_translation_unit,
)
from .tuning.inputs import (
    _read_shell_class_file,
    _requested_schedule_kinds,
    _requested_shell_class_names,
    _resolve_specifications,
)
from .tuning.manifest import (
    _validated_provenance,
    update_manifest_payload,
    write_tuned_manifest,
)
from .tuning.policy import (
    ScheduleTrial,
    _known_production_fock_subgroup_schedules,
    _production_fock_schedule_index,
    schedule_payload,
    supported_schedule_trials,
)
from .tuning.process import (
    _artifact_size,
    _compile_trial,
    _runtime_environment,
    _tool_version,
)
from .tuning.resources import _resource_rejections, estimate_occupancy

__all__ = [
    "ScheduleTrial",
    "StaticAlgebraModel",
    "_analysis_roots",
    "_artifact_size",
    "_balanced_component_labels",
    "_cached_static_algebra_model",
    "_class_name",
    "_compile_trial",
    "_identifier_suffix",
    "_integral_signature",
    "_isolate_schedule_symbols",
    "_known_production_fock_subgroup_schedules",
    "_oracle_schedule_trial",
    "_oracle_symbol_prefix",
    "_packed_force_geometry_analysis",
    "_production_fock_schedule_index",
    "_read_shell_class_file",
    "_requested_schedule_kinds",
    "_requested_shell_class_names",
    "_resolve_specifications",
    "_resource_rejections",
    "_run_autotune",
    "_runtime_environment",
    "_tool_version",
    "_validated_provenance",
    "argument_parser",
    "emit_schedule_driver",
    "emit_schedule_oracle_translation_unit",
    "emit_schedule_resource_translation_unit",
    "emit_schedule_translation_unit",
    "estimate_occupancy",
    "main",
    "schedule_payload",
    "static_algebra_model",
    "supported_schedule_trials",
    "update_manifest_payload",
    "write_tuned_manifest",
]

if __name__ == "__main__":
    raise SystemExit(main())
