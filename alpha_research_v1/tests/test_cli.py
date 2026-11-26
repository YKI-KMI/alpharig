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


def test_synth_reports_conditional_ic_for_interaction_panels(tmp_path: Path):
    """A sign-flipping panel must not be summarised by its pooled IC alone."""
    result = runner.invoke(
        app,
        [
            "synth",
            "--out",
            str(tmp_path / "lab"),
            "--stocks",
            "60",
            "--periods",
            "200",
            "--interaction",
            "--signal-ic",
            "0.06",
            "--seed",
            "5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "conditional IC" in result.output
    assert "calm" in result.output
    assert "high-vol" in result.output
    # The pooled IC is not the target for this panel and the output says so.
    assert "pooled IC is not the target" in result.output


def test_synth_output_does_not_point_at_a_command_that_does_not_exist(tmp_path: Path):
    result = runner.invoke(
        app,
        ["synth", "--out", str(tmp_path / "lab"), "--stocks", "40", "--periods", "120"],
    )
    assert result.exit_code == 0, result.output
    assert "alpharig audit" not in result.output
    assert "run `alpharig" not in result.output
