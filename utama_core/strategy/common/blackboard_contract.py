from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import py_trees
from py_trees.blackboard import Blackboard


@dataclass(frozen=True)
class BlackboardKeySpec:
    """Declared blackboard contract for a behaviour constructor key."""

    name: str
    access: str | py_trees.common.Access
    key_attr: str | None = None
    key: str | None = None
    required: bool = True
    type_name: str = "unknown"
    meaning: str = ""
    default: str | None = None
    bounds: tuple[float | None, float | None] | None = None

    def runtime_key(self, node: Any | None = None) -> str | None:
        if self.key is not None:
            return self.key
        if node is not None and self.key_attr is not None:
            if hasattr(node, self.key_attr):
                value = getattr(node, self.key_attr)
                return None if value is None else str(value)
        return self.name

    def to_dict(self, node: Any | None = None, *, resolve: bool = False) -> dict[str, Any]:
        key = self.runtime_key(node)
        payload = {
            "name": self.name,
            "key": key,
            "key_attr": self.key_attr,
            "access": _access_name(self.access),
            "required": self.required,
            "type": self.type_name,
            "meaning": self.meaning,
            "default": self.default,
            "bounds": self.bounds,
        }
        if resolve and node is not None and key is not None:
            payload["resolved_key"] = resolve_runtime_key(node, key)
        return payload


def declared_contract(target: Any, *, resolve: bool = False) -> list[dict[str, Any]]:
    return [spec.to_dict(target, resolve=resolve) for spec in _contract_specs(target)]


def resolve_runtime_key(node: Any, key: str) -> str:
    blackboards = getattr(node, "blackboards", ())
    for client in blackboards:
        try:
            return client.absolute_name(key)
        except KeyError:
            continue
    client = getattr(node, "blackboard", None)
    namespace = getattr(client, "namespace", None) if client is not None else None
    return Blackboard.absolute_name(namespace, key)


def register_blackboard_contract(node: Any) -> None:
    """Register a behaviour's declared blackboard keys on its py_trees client."""

    specs = _contract_specs(node)
    if not specs:
        return

    blackboard = getattr(node, "blackboard", None)
    if blackboard is None:
        raise RuntimeError(f"{_node_label(node)} has blackboard contracts but no attached blackboard client")

    for spec in specs:
        key = _registration_key(node, spec)
        if key is None:
            continue
        blackboard.register_key(key=key, access=_normalise_access(spec.access, node, spec))


def validate_blackboard_contracts(
    root: Any,
    *,
    seeded_keys: Iterable[str] | None = None,
    include_base_keys: bool = True,
) -> dict[str, Any]:
    """Validate declared read contracts against known writers and seeded inputs.

    This deliberately validates the declared contract graph, not py_trees runtime
    required keys. Many strategy keys are written by earlier behaviour nodes.
    """

    available_keys = {_comparison_key(key) for key in (seeded_keys or ())}
    if include_base_keys:
        from utama_core.strategy.common.base_blackboard import BaseBlackboard

        available_keys.update(BaseBlackboard.base_keys())

    reads: list[dict[str, Any]] = []
    writes: set[str] = set()
    errors: list[dict[str, Any]] = []

    for node in _iter_nodes(root):
        for spec in _contract_specs(node):
            try:
                key = _registration_key(node, spec)
                if key is None:
                    continue
                access = _normalise_access(spec.access, node, spec)
            except (TypeError, ValueError) as exc:
                errors.append(_validation_error("malformed_contract", node, spec, None, str(exc)))
                continue

            item = {
                "node": getattr(node, "name", node.__class__.__name__),
                "class_name": node.__class__.__name__,
                "name": spec.name,
                "key": key,
                "access": access.name,
            }
            if access == py_trees.common.Access.READ:
                reads.append(item)
            elif access in {py_trees.common.Access.WRITE, py_trees.common.Access.EXCLUSIVE_WRITE}:
                writes.add(_comparison_key(key))

    available_keys.update(writes)
    for item in reads:
        if _comparison_key(item["key"]) not in available_keys:
            errors.append(
                {
                    "code": "missing_writer",
                    **item,
                    "message": f"Required READ key {item['key']!r} has no declared writer or seeded input.",
                }
            )

    return {
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "read_keys": sorted({_comparison_key(item["key"]) for item in reads}),
        "write_keys": sorted(writes),
        "available_keys": sorted(available_keys),
    }


def _contract_specs(target: Any) -> tuple[BlackboardKeySpec, ...]:
    specs = getattr(target, "BLACKBOARD_CONTRACT", ())
    if specs is None:
        return ()
    if not isinstance(specs, tuple):
        raise TypeError(f"{_node_label(target)} BLACKBOARD_CONTRACT must be a tuple")
    for spec in specs:
        if not isinstance(spec, BlackboardKeySpec):
            raise TypeError(f"{_node_label(target)} BLACKBOARD_CONTRACT entries must be BlackboardKeySpec")
    return specs


def _registration_key(node: Any, spec: BlackboardKeySpec) -> str | None:
    if spec.key is not None:
        key = spec.key
    elif spec.key_attr is not None:
        if not hasattr(node, spec.key_attr):
            if spec.required:
                raise ValueError(
                    f"{_node_label(node)} contract {spec.name!r} references missing attribute {spec.key_attr!r}"
                )
            return None
        value = getattr(node, spec.key_attr)
        if value is None:
            if spec.required:
                raise ValueError(f"{_node_label(node)} contract {spec.name!r} resolved to None")
            return None
        key = str(value)
    else:
        key = spec.name

    if key == "":
        if spec.required:
            raise ValueError(f"{_node_label(node)} contract {spec.name!r} resolved to an empty key")
        return None
    return key


def _normalise_access(access: Any, node: Any, spec: BlackboardKeySpec) -> py_trees.common.Access:
    if isinstance(access, py_trees.common.Access):
        return access
    if isinstance(access, str):
        try:
            return py_trees.common.Access[access]
        except KeyError as exc:
            valid = ", ".join(item.name for item in py_trees.common.Access)
            raise ValueError(
                f"{_node_label(node)} contract {spec.name!r} has invalid access {access!r}; expected one of {valid}"
            ) from exc
    raise TypeError(f"{_node_label(node)} contract {spec.name!r} access must be a string or py_trees Access")


def _iter_nodes(root: Any) -> Iterable[Any]:
    root = getattr(root, "root", root)
    if hasattr(root, "iterate"):
        yield from root.iterate()
    else:
        yield root


def _validation_error(
    code: str,
    node: Any,
    spec: BlackboardKeySpec,
    key: str | None,
    message: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "node": getattr(node, "name", node.__class__.__name__),
        "class_name": node.__class__.__name__,
        "name": spec.name,
        "key": key,
        "access": _access_name(spec.access),
        "message": message,
    }


def _comparison_key(key: str) -> str:
    key = str(key)
    if key.startswith("/"):
        return key.rsplit("/", 1)[-1]
    return key


def _node_label(node: Any) -> str:
    return getattr(node, "name", getattr(node, "__name__", node.__class__.__name__))


def _access_name(access: Any) -> str:
    if hasattr(access, "name"):
        return access.name
    return str(access)
