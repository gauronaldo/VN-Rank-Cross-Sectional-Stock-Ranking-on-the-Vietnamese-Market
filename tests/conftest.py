"""Shared pytest fixtures."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@pytest.fixture
def default_config_path() -> Path:
    return DEFAULT_CONFIG


@pytest.fixture
def raw_config() -> dict[str, Any]:
    """A fresh, mutable copy of the default YAML as plain data."""
    data: dict[str, Any] = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    return copy.deepcopy(data)
