"""Strict JSON Schema conversion + null-as-absent output parsing (jettae.llm.schema)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from datetime import date, datetime
from typing import Annotated, Any, Literal

import pydantic
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, field_validator

from jettae.domain import (
    EdgeKind,
    EntityRef,
    ExtractedAttr,
    ExtractedEntity,
    ExtractedRelation,
    Extraction,
    Extractor,
    GroundingStatus,
    NodeKind,
    Perturbation,
    Scenario,
)
from jettae.llm.schema import (
    StrictSchemaError,
    drop_absent_nulls,
    parse_output,
    to_strict_json_schema,
    validate_output,
)

DOMAIN_MODELS: tuple[type[BaseModel], ...] = (
    ExtractedAttr,
    EntityRef,
    ExtractedEntity,
    ExtractedRelation,
    Extraction,
    GroundingStatus,
    Perturbation,
    Scenario,
)
#: the whole keyword vocabulary of a strict schema; $ref/title/default/minLength/... must all be gone
ALLOWED_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "anyOf",
        "enum",
        "const",
        "description",
        "format",
    }
)


# --------------------------------------------------------------------------- test models


class Cat(BaseModel):
    kind: Literal["cat"] = "cat"
    lives: int = Field(default=9, ge=0, le=9)
    name: str


class Dog(BaseModel):
    kind: Literal["dog"]
    name: str
    good: bool = True
    nickname: str | None = None


class Shelter(BaseModel):
    """Nested discriminated union plus optional nested models."""

    model_config = ConfigDict(extra="forbid")

    resident: Annotated[Cat | Dog, Field(discriminator="kind")]
    others: tuple[Annotated[Cat | Dog, Field(discriminator="kind")], ...] = ()
    backup: Cat | None = Field(default=None, description="optional backup cat")
    favourite: Cat = Field(default_factory=lambda: Cat(name="nabi"), description="field-level description")


class Constrained(BaseModel):
    code: str = Field(min_length=1, max_length=5, pattern=r"^[A-Z]+$", examples=["ABC"])
    score: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(min_length=1, max_length=3)
    at: datetime
    day: date
    link: AnyUrl


class Pair(BaseModel):
    point: tuple[int, int]
    mixed: tuple[int, str]


class Node(BaseModel):
    name: str
    children: tuple[Node, ...] = ()


class FreeForm(BaseModel):
    values: dict[str, int]


class Anything(BaseModel):
    payload: Any


class Open(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str


class Upper(BaseModel):
    word: str

    @field_validator("word")
    @classmethod
    def _upper(cls, value: str) -> str:
        if value != value.upper():
            raise ValueError("must be upper case")
        return value


# --------------------------------------------------------------------------- helpers


def walk(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Every schema node (not property-name maps) of a strict schema."""
    yield node
    for sub in node.get("properties", {}).values():
        yield from walk(sub)
    if "items" in node:
        yield from walk(node["items"])
    for branch in node.get("anyOf", ()):
        yield from walk(branch)


def object_nodes(schema: dict[str, Any]) -> list[dict[str, Any]]:
    return [node for node in walk(schema) if node.get("type") == "object"]


def allows_null(node: dict[str, Any]) -> bool:
    return node.get("type") == "null" or any(allows_null(branch) for branch in node.get("anyOf", ()))


_SCALAR_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def conforms(schema: dict[str, Any], value: Any) -> bool:
    """Minimal validator for the strict subset we emit (enough to check real dumps against it)."""
    if "anyOf" in schema:
        return any(conforms(branch, value) for branch in schema["anyOf"])
    if ("const" in schema and value != schema["const"]) or ("enum" in schema and value not in schema["enum"]):
        return False
    kind = schema["type"]
    if kind == "object":
        return (
            isinstance(value, dict)
            and set(value) == set(schema["required"]) == set(schema["properties"])
            and all(conforms(schema["properties"][key], item) for key, item in value.items())
        )
    if kind == "array":
        return isinstance(value, list) and all(conforms(schema["items"], item) for item in value)
    return _SCALAR_CHECKS[kind](value)


# --------------------------------------------------------------------------- strict schema shape


@pytest.mark.parametrize("model", DOMAIN_MODELS, ids=lambda m: m.__name__)
def test_domain_models_convert_to_strict_schemas(model: type[BaseModel]) -> None:
    schema = to_strict_json_schema(model)

    assert schema["type"] == "object"
    for node in object_nodes(schema):
        assert node["additionalProperties"] is False
        assert node["required"] == list(node["properties"])
    assert {key for node in walk(schema) for key in node} <= ALLOWED_KEYWORDS
    json.dumps(schema)  # serialisable as-is


def test_optional_fields_become_nullable_required_fields_do_not() -> None:
    props = to_strict_json_schema(ExtractedEntity)["properties"]

    assert allows_null(props["attrs"])  # tuple default () → nullable
    assert not allows_null(props["kind"])
    assert not allows_null(props["quote"])
    assert props["quote"]["description"] == "verbatim substring where the entity is mentioned"


def test_enum_const_and_description_are_kept() -> None:
    props = to_strict_json_schema(ExtractedRelation)["properties"]

    assert props["kind"]["enum"] == [kind.value for kind in EdgeKind]
    assert props["src"]["properties"]["kind"]["enum"] == [kind.value for kind in NodeKind]
    assert "verbatim" in props["quote"]["description"]


def test_extraction_root_inlines_nested_refs() -> None:
    schema = to_strict_json_schema(Extraction)
    entity_items = schema["properties"]["entities"]["anyOf"][0]["items"]
    attr_items = entity_items["properties"]["attrs"]["anyOf"][0]["items"]

    assert entity_items["properties"]["kind"]["enum"][0] == "service"
    assert attr_items["required"] == ["key", "value", "quote"]
    assert schema["properties"]["chunk_end"] == {"anyOf": [{"type": "integer"}, {"type": "null"}]}


def test_discriminated_union_becomes_anyof_with_mandatory_tags() -> None:
    schema = to_strict_json_schema(Shelter)
    resident = schema["properties"]["resident"]

    assert [branch["properties"]["kind"] for branch in resident["anyOf"]] == [
        {"type": "string", "const": "cat"},
        {"type": "string", "const": "dog"},
    ]
    # Cat.kind has a default, but a single-valued tag is never made nullable.
    assert not allows_null(resident["anyOf"][0]["properties"]["kind"])
    assert allows_null(resident["anyOf"][1]["properties"]["nickname"])
    others_items = schema["properties"]["others"]["anyOf"][0]["items"]
    assert len(others_items["anyOf"]) == 2


def test_field_description_overrides_model_docstring_and_wraps_nullable() -> None:
    props = to_strict_json_schema(Shelter)["properties"]

    assert props["favourite"]["description"] == "field-level description"
    assert props["favourite"]["anyOf"][-1] == {"type": "null"}
    assert props["backup"]["description"] == "optional backup cat"
    assert props["backup"]["anyOf"][-1] == {"type": "null"}
    assert to_strict_json_schema(Shelter)["description"].startswith("Nested discriminated union")


def test_unsupported_keywords_are_stripped_supported_formats_kept() -> None:
    props = to_strict_json_schema(Constrained)["properties"]

    assert props["code"] == {"type": "string"}
    assert props["score"] == {"type": "number"}
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["at"] == {"type": "string", "format": "date-time"}
    assert props["day"] == {"type": "string", "format": "date"}
    assert props["link"] == {"type": "string"}  # "uri" is not in the cross-provider subset


def test_stripped_constraints_are_still_enforced_on_parse() -> None:
    payload = {"code": "abc", "score": 2.0, "tags": [], "at": "2026-01-01T00:00:00Z", "day": "2026-01-01"}
    payload["link"] = "https://example.com"

    with pytest.raises(pydantic.ValidationError) as caught:
        parse_output(Constrained, json.dumps(payload))

    failed = {error["loc"][0] for error in caught.value.errors()}
    assert failed == {"code", "score", "tags"}


def test_positional_tuples_loosen_to_unions_and_arity_is_validated() -> None:
    props = to_strict_json_schema(Pair)["properties"]

    assert props["point"] == {"type": "array", "items": {"type": "integer"}}
    assert props["mixed"] == {"type": "array", "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]}}
    assert parse_output(Pair, '{"point": [1, 2], "mixed": [3, "x"]}') == Pair(point=(1, 2), mixed=(3, "x"))
    with pytest.raises(pydantic.ValidationError):
        parse_output(Pair, '{"point": [1, 2, 3], "mixed": [3, "x"]}')


@pytest.mark.parametrize(
    ("model", "fragment"),
    [(Node, "재귀"), (FreeForm, "dict"), (Anything, "Any")],
    ids=["recursive", "free-form-dict", "any"],
)
def test_unrepresentable_models_fail_loudly(model: type[BaseModel], fragment: str) -> None:
    with pytest.raises(StrictSchemaError, match=fragment):
        to_strict_json_schema(model)


def test_extra_allow_models_still_get_closed_objects() -> None:
    schema = to_strict_json_schema(Open)

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["name"]


def test_result_is_an_independent_copy() -> None:
    first = to_strict_json_schema(ExtractedEntity)
    first["properties"]["kind"]["enum"].append("hacked")
    first["required"].clear()

    second = to_strict_json_schema(ExtractedEntity)
    assert "hacked" not in second["properties"]["kind"]["enum"]
    assert second["required"] == ["kind", "name", "quote", "attrs"]


# --------------------------------------------------------------------------- parsing


def test_null_for_defaulted_field_means_absent() -> None:
    entity = parse_output(
        ExtractedEntity, json.dumps({"kind": "person", "name": "김대리", "quote": "김대리", "attrs": None})
    )

    assert entity == ExtractedEntity(kind=NodeKind.PERSON, name="김대리", quote="김대리")


def test_null_is_kept_where_the_field_itself_is_nullable() -> None:
    payload = {
        "id": "s1",
        "title": "결제 장애",
        "description": None,  # str = "" → absent → default
        "service_id": "service:payment",
        "incident_type": None,
        "perturbations": None,
        "deadline_minutes": None,  # float | None → real None
        "origin": None,
    }

    scenario = parse_output(Scenario, json.dumps(payload))

    assert scenario.description == ""
    assert scenario.deadline_minutes is None
    assert scenario.perturbations == ()


def test_nulls_are_pruned_inside_discriminated_union_branches() -> None:
    payload = {
        "resident": {"kind": "dog", "name": "baduk", "good": None, "nickname": None},
        "others": [{"kind": "cat", "lives": None, "name": "nabi"}],
        "backup": None,
        "favourite": None,
    }

    shelter = parse_output(Shelter, json.dumps(payload))

    assert shelter.resident == Dog(kind="dog", name="baduk", good=True, nickname=None)
    assert shelter.others == (Cat(name="nabi", lives=9),)
    assert shelter.backup is None
    assert shelter.favourite == Cat(name="nabi")


def test_required_null_is_not_hidden_by_pruning() -> None:
    with pytest.raises(pydantic.ValidationError):
        parse_output(
            ExtractedEntity, json.dumps({"kind": "person", "name": None, "quote": "q", "attrs": None})
        )


def test_invalid_json_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Expecting"):
        parse_output(ExtractedEntity, "not json")


def test_validators_run_after_parsing() -> None:
    assert parse_output(Upper, '{"word": "OK"}') == Upper(word="OK")
    with pytest.raises(pydantic.ValidationError):
        parse_output(Upper, '{"word": "no"}')


def test_validate_output_accepts_python_payloads() -> None:
    attr = validate_output(ExtractedAttr, {"key": "after_minutes", "value": 10, "quote": "10분"})

    assert attr.value == 10


def test_unknown_keys_are_left_for_validation_to_reject() -> None:
    payload = {"kind": "person", "name": "a", "quote": "a", "attrs": None, "extra": None}

    assert drop_absent_nulls(ExtractedEntity, payload) == {
        "kind": "person",
        "name": "a",
        "quote": "a",
        "extra": None,
    }
    with pytest.raises(pydantic.ValidationError):
        validate_output(ExtractedEntity, payload)


# --------------------------------------------------------------------------- properties

_text = st.text(alphabet=st.characters(codec="utf-8"), min_size=1, max_size=12)
_attr_values = st.one_of(
    _text,
    st.integers(min_value=-(10**9), max_value=10**9),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.lists(_text, max_size=3),
)
_attrs = st.lists(st.builds(ExtractedAttr, key=_text, value=_attr_values, quote=_text), max_size=3).map(tuple)
_refs = st.builds(EntityRef, kind=st.sampled_from(NodeKind), name=_text)
_entities = st.builds(ExtractedEntity, kind=st.sampled_from(NodeKind), name=_text, quote=_text, attrs=_attrs)
_relations = st.builds(
    ExtractedRelation, kind=st.sampled_from(EdgeKind), src=_refs, dst=_refs, quote=_text, attrs=_attrs
)
_extractions = st.builds(
    Extraction,
    doc_id=_text,
    extractor=st.sampled_from(Extractor),
    entities=st.lists(_entities, max_size=3).map(tuple),
    relations=st.lists(_relations, max_size=3).map(tuple),
    chunk_start=st.integers(min_value=0, max_value=10**6),
    chunk_end=st.none() | st.integers(min_value=0, max_value=10**6),
)


@settings(max_examples=60, deadline=None)
@given(_extractions)
def test_property_real_outputs_conform_and_round_trip(extraction: Extraction) -> None:
    payload = extraction.model_dump(mode="json")

    assert conforms(to_strict_json_schema(Extraction), payload)
    assert parse_output(Extraction, json.dumps(payload, ensure_ascii=False)) == extraction


@settings(max_examples=60, deadline=None)
@given(_entities, st.data())
def test_property_null_defaults_restore_defaults(entity: ExtractedEntity, data: st.DataObject) -> None:
    """A model answering "absent" (null) for defaulted fields yields the same object as omitting them."""
    payload = entity.model_dump(mode="json")
    if not payload["attrs"] and data.draw(st.booleans()):
        payload["attrs"] = None

    assert conforms(to_strict_json_schema(ExtractedEntity), payload)
    assert parse_output(ExtractedEntity, json.dumps(payload, ensure_ascii=False)) == entity


@settings(max_examples=40, deadline=None)
@given(st.sampled_from(DOMAIN_MODELS))
def test_property_conversion_is_deterministic(model: type[BaseModel]) -> None:
    assert json.dumps(to_strict_json_schema(model)) == json.dumps(to_strict_json_schema(model))


# --------------------------------------------------------------------------- review: edge shapes


class Empty(BaseModel):
    pass


class KeywordNamed(BaseModel):
    """Field names that collide with JSON Schema keywords must stay ordinary property names."""

    title: str = Field(default="제목 없음", description="문서 제목")
    type: Literal["장애", "점검"] = "장애"
    properties: list[str] | None = None
    description: str
    enum: int = 0


def test_empty_model_is_a_closed_empty_object() -> None:
    assert to_strict_json_schema(Empty) == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    assert parse_output(Empty, "{}") == Empty()
    with pytest.raises(pydantic.ValidationError):
        parse_output(Empty, "[]")


def test_keyword_named_fields_are_properties_not_keywords() -> None:
    schema = to_strict_json_schema(KeywordNamed)

    assert schema["required"] == ["title", "type", "properties", "description", "enum"]
    assert schema["properties"]["title"] == {
        "anyOf": [{"type": "string"}, {"type": "null"}],
        "description": "문서 제목",
    }
    assert schema["properties"]["type"]["anyOf"][0] == {"type": "string", "enum": ["장애", "점검"]}
    assert schema["description"].startswith("Field names that collide")
    parsed = parse_output(
        KeywordNamed,
        json.dumps(
            {"title": None, "type": "점검", "properties": None, "description": "설명", "enum": None},
            ensure_ascii=False,
        ),
    )
    assert parsed == KeywordNamed(type="점검", description="설명")


@pytest.mark.parametrize("model", [*DOMAIN_MODELS, Shelter, KeywordNamed], ids=lambda m: m.__name__)
def test_no_reference_or_unsupported_keyword_survives(model: type[BaseModel]) -> None:
    schema = to_strict_json_schema(model)

    # Keyword positions only — a property *named* "title" (Scenario, KeywordNamed) is legitimate.
    assert {key for node in walk(schema) for key in node} <= ALLOWED_KEYWORDS
    for node in object_nodes(schema):
        assert node["additionalProperties"] is False
        assert node["required"] == list(node["properties"])


@pytest.mark.parametrize("text", ["[]", "null", '"김대리"', "3"], ids=["array", "null", "string", "number"])
def test_non_object_replies_are_validation_errors(text: str) -> None:
    with pytest.raises(pydantic.ValidationError):
        parse_output(ExtractedAttr, text)


def test_pathological_nesting_is_a_value_error_not_a_crash() -> None:
    deep = '{"key": ' * 20_000 + "1" + "}" * 20_000

    with pytest.raises(ValueError, match="JSON 중첩"):
        parse_output(ExtractedAttr, deep)


def test_unicode_escaped_hangul_parses_to_the_same_text() -> None:
    attr = parse_output(ExtractedAttr, r'{"key": "tier", "value": "김대리", "quote": "10분"}')

    assert (attr.value, attr.quote) == ("김대리", "10분")


# --------------------------------------------------------------------------- review 2: non-JSON numbers (P3)


class Deadline(BaseModel):
    minutes: float | None = None


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e400", "-1e400"])
def test_non_finite_numbers_are_repairable_errors(literal: str) -> None:
    """``json.loads`` accepts these, Pydantic float fields would too — and a NaN deadline makes
    every comparison silently false. They are not JSON, so the reply goes back for repair."""
    with pytest.raises(ValueError, match=re.escape(literal)) as caught:
        parse_output(Deadline, f'{{"minutes": {literal}}}')

    assert not isinstance(caught.value, pydantic.ValidationError)


def test_non_finite_numbers_are_rejected_inside_nested_payloads() -> None:
    with pytest.raises(ValueError, match="NaN"):
        parse_output(ExtractedAttr, '{"key": "after_minutes", "value": [1, NaN], "quote": "10분"}')


@pytest.mark.parametrize(
    ("literal", "expected"),
    [("1.5", 1.5), ("-0.0", -0.0), ("1e300", 1e300), ("10", 10.0), ("null", None)],
)
def test_finite_numbers_still_parse(literal: str, expected: float | None) -> None:
    assert parse_output(Deadline, f'{{"minutes": {literal}}}').minutes == expected


@settings(max_examples=60, deadline=None)
@given(st.floats(allow_nan=False, allow_infinity=False))
def test_property_every_finite_float_round_trips(value: float) -> None:
    assert parse_output(Deadline, json.dumps({"minutes": value})).minutes == value


_json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=8)
    | st.sampled_from(["cat", "dog", "person", "김대리"]),
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(
            st.sampled_from(["kind", "name", "quote", "attrs", "resident", "others", "lives", "extra"]),
            children,
            max_size=5,
        )
    ),
    max_leaves=20,
)
_replies = st.one_of(
    _json_values.map(lambda value: json.dumps(value, ensure_ascii=False)),
    st.text(max_size=40),
    st.sampled_from(["NaN", '{"lives": 1e999}', "[" * 3000, '{"name": "\ud800"}', "1" * 5000]),
)


@settings(max_examples=150, deadline=None)
@given(st.sampled_from([ExtractedEntity, Shelter, Extraction, Scenario]), _replies)
def test_property_any_reply_is_parsed_or_rejected_as_repairable(model: type[BaseModel], reply: str) -> None:
    """Adapters repair on ``ValueError`` only; anything else would crash the caller's fallback."""
    try:
        parsed = parse_output(model, reply)
    except ValueError:
        return
    assert isinstance(parsed, model)
