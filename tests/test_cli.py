"""Tests for the command-line entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

from vn_rank.cli import main


def test_validate_config_succeeds_on_default(default_config_path: Path) -> None:
    assert main(["--config", str(default_config_path), "validate-config"]) == 0


def test_validate_config_fails_on_invalid_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("project:\n  seed: 1\n", encoding="utf-8")
    assert main(["--config", str(bad), "validate-config"]) == 1


def test_command_is_required() -> None:
    with pytest.raises(SystemExit):
        main([])
