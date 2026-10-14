"""Strict JSON Schema for schema-forced LLM output (P5: the model may only emit our schema).

Why a dedicated converter instead of handing ``model_json_schema()`` to the provider:

* Constrained decoding (Anthropic ``output_config.format``, OpenAI/vLLM ``strict`` json_schema)
  accepts only a subset of JSON Schema: every object must say ``additionalProperties: false``,
  every property must be ``required``, and numeric/string/array constraints, titles and
  defaults are rejected or ignored. Pydantic emits all of those.
* ``$ref``/``$defs`` support differs between providers and vLLM backends, so we inline them —
  one schema works everywhere and the cassette key (``replay``) is a pure function of it.
* "Every property required" would make optional fields mandatory, so optional/defaulted fields
  become *nullable* instead. :func:`parse_output` maps such a ``null`` back to "absent" before
  validation, so the field's default applies — an absent fact stays absent (P2) instead of the
  model being forced to invent a value for it.

Constraints we strip (``minLength``, ``maximum`` …) are still enforced afterwards, because the
output is always validated by the original Pydantic model.
"""

from __future__ import annotations

import copy
import functools
import json
import math
from collections.abc import Mapping
from typing import Any, Final, NoReturn, TypeVar

from pydantic import BaseModel

_T = TypeVar("_T", bound=BaseModel)

JsonSchema = dict[str, Any]

#: keywords that survive strictification — a whitelist, so unknown keywords never leak through
_KEPT_SCALAR_KEYWORDS: Final = ("type", "description", "enum", "const")
_KEPT_FORMATS: Final = frozenset({"date-time", "date"})
_DEFS_PREFIX: Final = "#/$defs/"

#: keywords whose value is a single subschema / list of subschemas / map of subschemas
_SUBSCHEMA_KEYS: Final = frozenset({"items", "additionalProperties", "not", "contains", "if", "then", "else"})
_SUBSCHEMA_LIST_KEYS: Final = frozenset({"anyOf", "oneOf", "prefixItems"})
_SUBSCHEMA_MAP_KEYS: Final = frozenset({"properties", "patternProperties", "dependentSchemas"})

_TYPE_MATCHERS: Final[Mapping[str, type | tuple[type, ...]]] = {
    "null": type(None),
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


class StrictSchemaError(ValueError):
    """The Pydantic model cannot be expressed as a strict structured-output schema."""


# --------------------------------------------------------------------------- public API


def to_strict_json_schema(model: type[BaseModel]) -> JsonSchema:
    """Return the provider-neutral strict JSON Schema for ``model``.

    The result is a fresh deep copy (callers may mutate it); the computation is cached per class.
    Raises :class:`StrictSchemaError` for shapes strict decoding cannot express (recursive
    models, free-form ``dict``/``Any`` fields, non-object roots) — failing loudly at the first
    call beats silently dropping data the model was never allowed to emit.
    """
    return copy.deepcopy(_strict_schema_cached(model))


def parse_output(schema: type[_T], text: str) -> _T:
    """Decode model output text and validate it against ``schema`` in JSON mode.

    Raises ``ValueError`` subclasses only (``json.JSONDecodeError``, ``pydantic.ValidationError``
    or plain ``ValueError`` for pathologically nested JSON) so adapters can treat every bad reply
    as "repairable" — and, after the one repair, as :class:`~jettae.llm.base.LLMOutputError`
    rather than an unexpected crash. Strict decoding bounds nesting by the schema, but a gateway
    that ignores ``response_format`` can return anything.

    Python's ``json`` also accepts ``NaN``/``Infinity`` and turns overflowing literals such as
    ``1e400`` into ``inf``. Neither is JSON, and a non-finite number would slip through
    Pydantic float fields into deadline arithmetic where every comparison is silently false —
    unknown masquerading as a result (P3) — so both are rejected as repairable errors.
    """
    try:
        decoded = json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float)
        canonical = json.dumps(drop_absent_nulls(schema, decoded), ensure_ascii=False)
    except RecursionError as exc:
        raise ValueError("JSON 중첩이 너무 깊습니다 / JSON nesting is too deep") from exc
    return schema.model_validate_json(canonical)


def validate_output(schema: type[_T], payload: Any) -> _T:
    """Validate an already-decoded Python payload (fake handlers, tests) against ``schema``."""
    return schema.model_validate(drop_absent_nulls(schema, payload))


def drop_absent_nulls(schema: type[BaseModel], payload: Any) -> Any:
    """Remove ``null`` values that strict mode forced onto defaulted, non-nullable fields.

    The strict schema makes optional fields nullable; Pydantic would reject ``None`` for a field
    like ``attrs: tuple[...] = ()``. Dropping the key restores the declared default. Fields whose
    own type admits ``None`` keep their ``null`` — there it is a real value.
    """
    return _prune(_resolved_schema(schema), payload)


# --------------------------------------------------------------------------- $ref inlining


@functools.lru_cache(maxsize=256)
def _resolved_schema(model: type[BaseModel]) -> JsonSchema:
    """Pydantic's schema with every ``$ref`` (and single-item ``allOf``) inlined."""
    raw = model.model_json_schema()
    resolved: JsonSchema = _inline(raw, raw.get("$defs", {}), ())
    return resolved


def _inline(node: Any, defs: Mapping[str, Any], stack: tuple[str, ...]) -> Any:
    if not isinstance(node, dict):
        return node
    out: JsonSchema = {}
    if "$ref" in node:
        out.update(_inline_ref(node["$ref"], defs, stack))
    all_of = node.get("allOf")
    if all_of is not None:
        if len(all_of) != 1:
            raise StrictSchemaError("여러 스키마를 합치는 allOf는 strict 스키마로 표현할 수 없습니다")
        out.update(_inline(all_of[0], defs, stack))
    for key, value in node.items():
        if key in ("$ref", "allOf", "$defs"):
            continue
        out[key] = _inline_keyword(key, value, defs, stack)
    return out


def _inline_ref(ref: str, defs: Mapping[str, Any], stack: tuple[str, ...]) -> JsonSchema:
    if not ref.startswith(_DEFS_PREFIX):
        raise StrictSchemaError(f"지원하지 않는 $ref 형식입니다: {ref}")
    name = ref.removeprefix(_DEFS_PREFIX)
    if name in stack:
        cycle = " -> ".join((*stack, name))
        raise StrictSchemaError(f"재귀 스키마는 strict 출력으로 표현할 수 없습니다: {cycle}")
    if name not in defs:
        raise StrictSchemaError(f"$defs에 없는 참조입니다: {ref}")
    target = _inline(defs[name], defs, (*stack, name))
    return dict(target)


def _inline_keyword(key: str, value: Any, defs: Mapping[str, Any], stack: tuple[str, ...]) -> Any:
    """Recurse only into keywords that hold schemas; literal data (enum, default …) is copied."""
    if key in _SUBSCHEMA_KEYS:
        return _inline(value, defs, stack)
    if key in _SUBSCHEMA_LIST_KEYS and isinstance(value, list):
        return [_inline(item, defs, stack) for item in value]
    if key in _SUBSCHEMA_MAP_KEYS and isinstance(value, dict):
        return {name: _inline(sub, defs, stack) for name, sub in value.items()}
    return value


# --------------------------------------------------------------------------- strictification


@functools.lru_cache(maxsize=256)
def _strict_schema_cached(model: type[BaseModel]) -> JsonSchema:
    strict = _strictify(_resolved_schema(model), model.__name__)
    if strict.get("type") != "object":
        raise StrictSchemaError(f"{model.__name__}: 최상위 스키마는 객체여야 합니다")
    return strict


def _strictify(node: Mapping[str, Any], path: str) -> JsonSchema:
    out: JsonSchema = {}
    branches = _union_branches(node)
    if branches:
        out["anyOf"] = [_strictify(branch, f"{path}|{i}") for i, branch in enumerate(branches)]
    for key in _KEPT_SCALAR_KEYWORDS:
        if key in node:
            out[key] = copy.deepcopy(node[key])
    if node.get("format") in _KEPT_FORMATS:
        out["format"] = node["format"]
    if _is_object(node):
        out.update(_strict_object(node, path))
    elif _is_array(node):
        out["items"] = _strict_items(node, path)
    if not any(key in out for key in ("type", "enum", "const", "anyOf")):
        raise StrictSchemaError(f"{path}: 타입이 지정되지 않은 값(Any 등)은 strict 스키마로 표현 불가")
    return out


def _strict_object(node: Mapping[str, Any], path: str) -> JsonSchema:
    properties = node.get("properties")
    if not isinstance(properties, dict):
        raise StrictSchemaError(
            f"{path}: 키가 정해지지 않은 객체(dict 등)는 strict 스키마로 표현 불가 — "
            "필드를 명시한 모델을 쓰세요"
        )
    required = set(node.get("required", ()))
    strict_props = {
        name: _strict_property(sub, required=name in required, path=f"{path}.{name}")
        for name, sub in properties.items()
    }
    return {
        "type": "object",
        "properties": strict_props,
        "required": list(strict_props),
        "additionalProperties": False,
    }


def _strict_items(node: Mapping[str, Any], path: str) -> JsonSchema:
    """Arrays keep one ``items`` schema; positional tuples are loosened to a union of positions.

    Strict decoding has no ``prefixItems``; Pydantic still enforces tuple arity afterwards and a
    violation goes through the adapter's one-shot repair.
    """
    candidates = [*node.get("prefixItems", ())]
    if isinstance(node.get("items"), dict):
        candidates.append(node["items"])
    if not candidates:
        raise StrictSchemaError(f"{path}: 원소 타입이 없는 배열은 strict 스키마로 표현할 수 없습니다")
    strict_items: list[JsonSchema] = []
    seen: set[str] = set()
    for i, candidate in enumerate(candidates):
        strict = _strictify(candidate, f"{path}[{i}]")
        fingerprint = json.dumps(strict, sort_keys=True, ensure_ascii=False)
        if fingerprint not in seen:
            seen.add(fingerprint)
            strict_items.append(strict)
    return strict_items[0] if len(strict_items) == 1 else {"anyOf": strict_items}


def _strict_property(sub: Mapping[str, Any], *, required: bool, path: str) -> JsonSchema:
    strict = _strictify(sub, path)
    if required or _allows_null(strict) or _is_single_valued(strict):
        # A single-valued (const) field gains nothing from being nullable, and keeping it
        # mandatory preserves discriminator tags of tagged unions.
        return strict
    return _nullable(strict)


def _nullable(strict: JsonSchema) -> JsonSchema:
    description = strict.get("description")
    body = {key: value for key, value in strict.items() if key != "description"}
    if set(body) == {"anyOf"}:
        out: JsonSchema = {"anyOf": [*body["anyOf"], {"type": "null"}]}
    else:
        out = {"anyOf": [body, {"type": "null"}]}
    if description is not None:
        out["description"] = description
    return out


# --------------------------------------------------------------------------- schema predicates


def _union_branches(node: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """``anyOf`` and ``oneOf`` (Pydantic's discriminated unions) are treated alike."""
    return [*node.get("anyOf", ()), *node.get("oneOf", ())]


def _is_object(node: Mapping[str, Any]) -> bool:
    return node.get("type") == "object" or "properties" in node


def _is_array(node: Mapping[str, Any]) -> bool:
    return node.get("type") == "array" or "prefixItems" in node


def _is_single_valued(node: Mapping[str, Any]) -> bool:
    enum = node.get("enum")
    return "const" in node or (isinstance(enum, list) and len(enum) == 1)


def _is_unconstrained(node: Mapping[str, Any]) -> bool:
    constraining = ("type", "enum", "const", "anyOf", "oneOf", "properties", "items", "prefixItems")
    return not any(key in node for key in constraining)


def _allows_null(node: Mapping[str, Any]) -> bool:
    node_type = node.get("type")
    if node_type == "null" or (isinstance(node_type, list) and "null" in node_type):
        return True
    if "const" in node:
        return node["const"] is None
    if "enum" in node:
        return None in node["enum"]
    branches = _union_branches(node)
    if branches:
        return any(_allows_null(branch) for branch in branches)
    return _is_unconstrained(node)


# --------------------------------------------------------------------------- null pruning


def _prune(node: Mapping[str, Any], value: Any) -> Any:
    branches = _union_branches(node)
    if branches:
        return _prune_union(branches, value)
    if isinstance(value, dict):
        return _prune_object(node, value)
    if isinstance(value, list):
        return _prune_array(node, value)
    return value


def _prune_union(branches: list[Mapping[str, Any]], value: Any) -> Any:
    """Prune along the branch(es) the value fits; ambiguous disagreement leaves it untouched."""
    results = [_prune(branch, value) for branch in branches if _shallow_match(branch, value)]
    if results and all(result == results[0] for result in results):
        return results[0]
    return value


def _prune_object(node: Mapping[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    properties = node.get("properties")
    if not isinstance(properties, dict):
        extra = node.get("additionalProperties")
        if isinstance(extra, dict):
            return {key: _prune(extra, item) for key, item in value.items()}
        return value
    required = set(node.get("required", ()))
    out: dict[str, Any] = {}
    for key, item in value.items():
        sub = properties.get(key)
        if sub is None:
            out[key] = item
        elif item is None and key not in required and not _allows_null(sub):
            continue
        else:
            out[key] = _prune(sub, item)
    return out


def _prune_array(node: Mapping[str, Any], value: list[Any]) -> list[Any]:
    prefix = node.get("prefixItems") or []
    items = node.get("items")
    out: list[Any] = []
    for index, item in enumerate(value):
        sub = prefix[index] if index < len(prefix) else items
        out.append(_prune(sub, item) if isinstance(sub, dict) else item)
    return out


def _shallow_match(node: Mapping[str, Any], value: Any) -> bool:
    """Cheap structural check used only to pick union branches for pruning."""
    if "const" in node:
        return _same_json_value(value, node["const"])
    if "enum" in node:
        return any(_same_json_value(value, option) for option in node["enum"])
    branches = _union_branches(node)
    if branches:
        return any(_shallow_match(branch, value) for branch in branches)
    return _type_matches(node, value) and _object_keys_match(node, value)


def _type_matches(node: Mapping[str, Any], value: Any) -> bool:
    node_type = node.get("type")
    if isinstance(node_type, str):
        return _matches_type(node_type, value)
    return isinstance(value, dict) or not _is_object(node)


def _object_keys_match(node: Mapping[str, Any], value: Any) -> bool:
    """Keys must be declared and tag-like (const/enum) properties must agree.

    This is what tells the variants of a discriminated union apart.
    """
    properties = node.get("properties")
    if not (isinstance(value, dict) and isinstance(properties, dict)):
        return True
    if not set(value) <= set(properties):
        return False
    return all(
        _shallow_match(properties[key], item)
        for key, item in value.items()
        if "const" in properties[key] or "enum" in properties[key]
    )


def _matches_type(node_type: str, value: Any) -> bool:
    expected = _TYPE_MATCHERS.get(node_type)
    if expected is None:
        return True
    if isinstance(value, bool) and node_type != "boolean":
        return False
    return isinstance(value, expected)


def _same_json_value(left: Any, right: Any) -> bool:
    """JSON equality: ``True`` is not ``1``."""
    return isinstance(left, bool) == isinstance(right, bool) and left == right


# --------------------------------------------------------------------------- JSON number hooks


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"JSON 표준에 없는 값({name})입니다 / {name} is not valid JSON")


def _finite_float(literal: str) -> float:
    value = float(literal)
    if not math.isfinite(value):
        raise ValueError(f"표현할 수 있는 범위를 넘는 수({literal[:32]})입니다 / number out of range")
    return value
