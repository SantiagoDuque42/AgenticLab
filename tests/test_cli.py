"""The CLI refuses to start on native Windows and points to WSL."""

import pytest
import typer
from typer.testing import CliRunner

from miniagent import cli

runner = CliRunner()


def test_refuses_native_windows(monkeypatch):
    monkeypatch.setattr(cli.os, "name", "nt")
    with pytest.raises(typer.Exit) as exc:
        cli.require_posix()
    assert exc.value.exit_code == 2


def test_every_command_is_guarded(monkeypatch):
    monkeypatch.setattr(cli, "require_posix", lambda: (_ for _ in ()).throw(typer.Exit(2)))
    result = runner.invoke(cli.app, ["tasks"])
    assert result.exit_code == 2


def test_posix_runs_normally():
    result = runner.invoke(cli.app, ["tasks"])
    assert result.exit_code == 0
    assert "fix-off-by-one" in result.output
