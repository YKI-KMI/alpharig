"""CLI smoke tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from alpharig.cli import app
from alpharig.synth.generator import PanelSpec

runner = CliRunner()


def test_synth_command_writes_a_panel(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "synth",
            "--out",
            str(tmp_path / "lab"),
            "--stocks",
            "40",
            "--periods",
            "120",
            "--seed",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "lab" / "prices.parquet").exists()
    assert (tmp_path / "lab" / "truth.json").exists()
    assert "n_eff" in result.output
    assert "RESEARCH ONLY" in result.output


def test_synth_command_accepts_yaml_config(tmp_path: Path):
    config = Path(__file__).resolve().parents[1] / "configs" / "synth_lab.yaml"
    assert PanelSpec.from_yaml(config).n_stocks == 200
    result = runner.invoke(
        app,
        [
            "synth",
            "--config",
            str(config),
            "--out",
            str(tmp_path / "lab"),
            "--stocks",
            "30",
            "--periods",
            "90",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "30 names" in result.output


def test_stages_command_lists_the_plan():
    result = runner.invoke(app, ["stages"])
    assert result.exit_code == 0
    assert "data-layer" in result.output
    assert "validation-audit" in result.output
