"""Tests for strict dataclass construction from plain data."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import pytest

from vn_rank.utils.dataclass_parsing import SchemaError, build_dataclass, to_plain


@dataclasses.dataclass(frozen=True)
class Inner:
    when: dt.date
    where: Path


@dataclasses.dataclass(frozen=True)
class Outer:
    count: int
    ratio: float
    label: str | None
    windows: tuple[int, ...]
    limits: dict[str, float]
    inner: Inner


def _raw() -> dict[str, object]:
    return {
        "count": 3,
        "ratio": 1,
        "label": None,
        "windows": [1, 2],
        "limits": {"A": 0.5},
        "inner": {"when": dt.date(2024, 1, 2), "where": "a/b"},
    }


def test_builds_nested_dataclass_with_conversions() -> None:
    built = build_dataclass(Outer, _raw(), "root")
    assert built == Outer(3, 1.0, None, (1, 2), {"A": 0.5}, Inner(dt.date(2024, 1, 2), Path("a/b")))
    assert isinstance(built.ratio, float)


def test_to_plain_inverts_build() -> None:
    built = build_dataclass(Outer, _raw(), "root")
    assert build_dataclass(Outer, to_plain(dataclasses.asdict(built)), "root") == built


def test_error_message_names_the_field() -> None:
    raw = _raw()
    raw["inner"] = {"when": dt.datetime(2024, 1, 2, 9, 0), "where": "a"}
    with pytest.raises(SchemaError, match=r"root\.inner\.when: expected date"):
        build_dataclass(Outer, raw, "root")


def test_tuple_items_are_type_checked() -> None:
    raw = _raw()
    raw["windows"] = [1, 2.5]
    with pytest.raises(SchemaError, match=r"root\.windows\[1\]: expected int"):
        build_dataclass(Outer, raw, "root")


def test_empty_path_is_rejected() -> None:
    raw = _raw()
    raw["inner"] = {"when": dt.date(2024, 1, 2), "where": ""}
    with pytest.raises(SchemaError, match="expected Path"):
        build_dataclass(Outer, raw, "root")
