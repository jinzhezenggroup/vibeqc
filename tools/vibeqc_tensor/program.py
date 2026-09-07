"""Deterministic tensor programs, logical hashes, and data-only replay.

Provenance is preserved in artifacts but excluded from equation identity.
Tensor/space semantics, exact factors, input roles, and primitive versions
are included. No serialized string is interpreted as Python code.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .ir import Node
from .types import spec_from_payload, spec_to_payload

SCHEMA = "vibeqc.tensor"
VERSION = 1
PRIMITIVE_VERSION = 1
CONVENTIONS = {
    "indexing": "zero-based; half-open ranges; ordered gather coordinates",
    "reshape": "logical C order; physical copies/views are executor decisions",
    "inner_product": "real Euclidean dense; packed coordinates carry orbit weights",
    "coefficient": "reduced rational numerator/positive denominator",
    "aliasing": "immutable SSA; no in-place writes",
}


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _topological(roots) -> tuple[Node, ...]:
    """Iterative DFS handles deep equations without recursive hashing."""
    order, states = [], {}
    for root in roots:
        stack = [(root, False)]
        while stack:
            node, finish = stack.pop()
            if finish:
                states[node] = 2
                order.append(node)
            elif states.get(node) == 1:
                raise ValueError("tensor program must be acyclic")
            elif states.get(node) != 2:
                states[node] = 1
                stack.append((node, True))
                stack.extend((child, False) for child in reversed(node.inputs))
    return tuple(order)


def node_hashes(nodes) -> dict[Node, str]:
    """Content addresses include full spin/symmetry semantics, not storage."""
    hashes = {}
    for node in nodes:
        hashes[node] = hash_node(node, hashes)
    return hashes


def hash_node(node: Node, input_hashes: Mapping[Node, str]) -> str:
    """Hash one definition using already computed dependency identities."""
    return _hash(
        {
            "primitive_version": PRIMITIVE_VERSION,
            "op": node.op,
            "spec": spec_to_payload(node.spec, logical=True),
            "attributes": node.attrs,
            "inputs": [input_hashes[n] for n in node.inputs],
        }
    )


@dataclass(frozen=True, init=False)
class Program:
    """Named outputs plus optional retained definitions and JSON provenance.

    Definitions allow an unoptimized mathematical artifact to retain dead
    work for inspection. Only output-reachable nodes are executed. The
    canonicalizer returns a new program and never erases the original DAG.
    """

    outputs: Mapping[str, Node]
    definitions: tuple[Node, ...]
    _provenance_json: str

    def __init__(self, outputs: Mapping[str, Node], definitions=(), provenance=None):
        if not isinstance(outputs, Mapping) or not outputs:
            raise ValueError("program requires named outputs")
        if any(not isinstance(k, str) or not k.isidentifier() for k in outputs):
            raise ValueError("output names must be identifiers")
        outputs = dict(sorted(outputs.items()))
        definitions = tuple(definitions)
        if any(not isinstance(n, Node) for n in (*outputs.values(), *definitions)):
            raise TypeError("program definitions and outputs must be tensor nodes")
        object.__setattr__(self, "outputs", MappingProxyType(outputs))
        object.__setattr__(self, "definitions", definitions)
        # Snapshot nested provenance so subsequent caller mutations cannot
        # silently change an already serialized or validated program.
        provenance = {} if provenance is None else provenance
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be a JSON object")
        object.__setattr__(self, "_provenance_json", _json(dict(provenance)))
        spaces, inputs = {}, {}
        for node in self.nodes:
            for index in node.spec.indices:
                previous = spaces.setdefault(index.space.name, index.space)
                if previous != index.space:
                    raise ValueError(
                        "space name has inconsistent population/spin definitions"
                    )
            if node.op == "input":
                name = node.attrs["name"]
                signature = spec_to_payload(node.spec, logical=True)
                if inputs.setdefault(name, signature) != signature:
                    raise ValueError(
                        "input name reused with incompatible tensor semantics"
                    )

    @property
    def provenance(self) -> dict:
        """Return a detached JSON snapshot, including nested lists/maps."""
        return json.loads(self._provenance_json)

    @property
    def nodes(self) -> tuple[Node, ...]:
        """Deterministic dependency order independent of builder/dict insertion."""
        nodes = _topological((*self.outputs.values(), *self.definitions))
        hashes, depth = node_hashes(nodes), {}
        for node in nodes:
            depth[node] = 1 + max((depth[n] for n in node.inputs), default=-1)
        return tuple(sorted(nodes, key=lambda n: (depth[n], hashes[n])))

    @property
    def live_nodes(self) -> tuple[Node, ...]:
        live = set(_topological(self.outputs.values()))
        return tuple(n for n in self.nodes if n in live)

    @property
    def logical_hash(self) -> str:
        """Identify output equations, excluding dead nodes and provenance."""
        hashes = node_hashes(self.nodes)
        return _hash(
            {
                "schema": SCHEMA,
                "version": VERSION,
                "primitive_version": PRIMITIVE_VERSION,
                "conventions": CONVENTIONS,
                "outputs": {name: hashes[node] for name, node in self.outputs.items()},
            }
        )

    @property
    def debug_names(self) -> dict[Node, str]:
        """Stable, source-safe content names with duplicate occurrence suffixes."""
        hashes = node_hashes(self.nodes)
        counts, result = {}, {}
        for node in self.nodes:
            digest = hashes[node]
            count = counts.get(digest, 0)
            counts[digest] = count + 1
            result[node] = f"n_{digest}_{count}"
        return result

    def to_payload(self) -> dict:
        """Persist replayable equations, conventions, hashes, and provenance."""
        names = self.debug_names
        return {
            "schema": SCHEMA,
            "schema_version": VERSION,
            "primitive_version": PRIMITIVE_VERSION,
            "conventions": dict(CONVENTIONS),
            "provenance": json.loads(self._provenance_json),
            "logical_hash": self.logical_hash,
            "nodes": [
                {
                    "id": names[n],
                    "op": n.op,
                    "spec": spec_to_payload(n.spec),
                    "inputs": [names[child] for child in n.inputs],
                    "attributes": n.attrs,
                }
                for n in self.nodes
            ],
            "outputs": {name: names[n] for name, n in self.outputs.items()},
        }

    def dumps(self) -> str:
        """Canonical JSON with a final newline, suitable for checked-in examples."""
        return (
            json.dumps(self.to_payload(), indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )

    @classmethod
    def from_payload(cls, payload: dict) -> Program:
        """Reject incompatible versions, forward references, and invalid nodes."""
        if not isinstance(payload, dict) or set(payload) != {
            "schema",
            "schema_version",
            "primitive_version",
            "conventions",
            "provenance",
            "logical_hash",
            "nodes",
            "outputs",
        }:
            raise ValueError("invalid tensor program fields")
        if (
            payload["schema"] != SCHEMA
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != VERSION
            or type(payload["primitive_version"]) is not int
            or payload["primitive_version"] != PRIMITIVE_VERSION
            or payload["conventions"] != CONVENTIONS
        ):
            raise ValueError(
                "unsupported tensor schema/primitive version or conventions"
            )
        nodes = {}
        try:
            for row in payload["nodes"]:
                if set(row) != {"id", "op", "spec", "inputs", "attributes"}:
                    raise ValueError("invalid tensor node fields")
                if not isinstance(row["id"], str) or row["id"] in nodes:
                    raise ValueError("duplicate or invalid node id")
                node = Node(
                    row["op"],
                    tuple(nodes[name] for name in row["inputs"]),
                    spec_from_payload(row["spec"]),
                    tuple(row["attributes"].items()),
                )
                nodes[row["id"]] = node
            program = cls(
                {name: nodes[ref] for name, ref in payload["outputs"].items()},
                tuple(nodes.values()),
                payload["provenance"],
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(
                "malformed tensor program or non-topological reference"
            ) from exc
        if program.logical_hash != payload["logical_hash"]:
            raise ValueError("tensor logical hash mismatch")
        names = program.debug_names
        if any(names[node] != name for name, node in nodes.items()):
            raise ValueError("noncanonical tensor debug/source name")
        return program

    @classmethod
    def loads(cls, source: str) -> Program:
        """Load JSON data, never executable expressions or pickled objects."""

        def reject_duplicates(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON field: {key}")
                result[key] = value
            return result

        return cls.from_payload(json.loads(source, object_pairs_hook=reject_duplicates))
