"""Strict construction of frozen dataclasses from plain YAML-style data.

Generic helper used by :mod:`vn_rank.config`. Unknown keys, missing keys, and values of the
wrong type raise :class:`SchemaError`. Supported field types: ``int``, ``float``, ``bool``,
``str``, :class:`pathlib.Path`, :class:`datetime.date`, ``tuple[X, ...]``, ``dict[K, V]``,
``X | None``, and nested dataclasses.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import types
import typing
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

T = TypeVar("T", bound="DataclassInstance")


class SchemaError(ValueError):
    """Raised when data does not match the dataclass schema."""


def to_plain(value: Any) -> Any:
    """Convert dataclass-derived data into YAML-safe plain data.

    Args:
        value: Output of :func:`dataclasses.asdict` or any nested part of it.

    Returns:
        The same data with tuples as lists and paths as POSIX strings.
    """
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [to_plain(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def build_dataclass(cls: type[T], raw: object, where: str) -> T:
    """Recursively build ``cls`` from nested mappings, checking keys and types.

    Args:
        cls: Target dataclass type; field annotations drive the conversion.
        raw: Nested mappings and lists, e.g. from ``yaml.safe_load``.
        where: Dotted location used in error messages, e.g. ``"config"``.

    Returns:
        An instance of ``cls``. YAML lists become tuples and strings become paths.

    Raises:
        SchemaError: On unknown or missing keys or values of the wrong type.
    """
    if not isinstance(raw, Mapping):
        raise SchemaError(f"{where}: expected a mapping, got {type(raw).__name__}")
    names = [field.name for field in dataclasses.fields(cls)]
    unknown = sorted(set(raw) - set(names))
    missing = sorted(set(names) - set(raw))
    if unknown or missing:
        raise SchemaError(f"{where}: unknown keys {unknown}, missing keys {missing}")
    hints = typing.get_type_hints(cls)
    kwargs = {name: _coerce(raw[name], hints[name], f"{where}.{name}") for name in names}
    return cls(**kwargs)


def _coerce(value: object, annotation: Any, where: str) -> Any:
    origin = typing.get_origin(annotation)
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return build_dataclass(annotation, value, where)
    handler = _ORIGIN_HANDLERS.get(origin) if origin is not None else None
    if handler is not None:
        return handler(value, annotation, where)
    return _coerce_scalar(value, annotation, where)


def _coerce_optional(value: object, annotation: Any, where: str) -> Any:
    args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
    if value is None:
        return None
    return _coerce(value, args[0], where)


def _coerce_tuple(value: object, annotation: Any, where: str) -> Any:
    if not isinstance(value, list):
        raise SchemaError(f"{where}: expected a list, got {type(value).__name__}")
    item_type = typing.get_args(annotation)[0]
    return tuple(_coerce(item, item_type, f"{where}[{i}]") for i, item in enumerate(value))


def _coerce_dict(value: object, annotation: Any, where: str) -> Any:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{where}: expected a mapping, got {type(value).__name__}")
    key_type, item_type = typing.get_args(annotation)
    return {
        _coerce(key, key_type, f"{where} key"): _coerce(item, item_type, f"{where}.{key}")
        for key, item in value.items()
    }


def _coerce_scalar(value: object, annotation: Any, where: str) -> Any:
    accepts, convert = _SCALAR_RULES[annotation]
    if not accepts(value):
        expected = getattr(annotation, "__name__", str(annotation))
        raise SchemaError(f"{where}: expected {expected}, got {value!r}")
    return convert(value)


def is_number(value: object) -> bool:
    """Return whether ``value`` is an int or float but not a bool."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_date(value: object) -> bool:
    return isinstance(value, dt.date) and not isinstance(value, dt.datetime)


_ORIGIN_HANDLERS: dict[Any, Callable[[object, Any, str], Any]] = {
    types.UnionType: _coerce_optional,
    typing.Union: _coerce_optional,
    tuple: _coerce_tuple,
    dict: _coerce_dict,
}

_SCALAR_RULES: dict[Any, tuple[Callable[[object], bool], Callable[[Any], Any]]] = {
    int: (lambda v: isinstance(v, int) and not isinstance(v, bool), int),
    float: (is_number, float),
    bool: (lambda v: isinstance(v, bool), bool),
    str: (lambda v: isinstance(v, str), str),
    Path: (lambda v: isinstance(v, str) and v != "", Path),
    dt.date: (_is_date, lambda v: v),
}
