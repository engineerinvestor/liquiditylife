"""Tests for the CLI interface."""

from click.testing import CliRunner

from liquiditylife.cli.main import cli


class TestCLI:
    def test_list_calibrations(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["list-calibrations"])
        assert result.exit_code == 0
        assert "adams_baseline" in result.output
        assert "adams_frictionless" in result.output
        assert "toy_demo_small_grid" in result.output

    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_solve_missing_calibration(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["solve"])
        assert result.exit_code != 0  # missing required --calibration

    def test_simulate_missing_calibration(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["simulate"])
        assert result.exit_code != 0

    def test_sweep_missing_args(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["sweep", "policy-surface"])
        assert result.exit_code != 0  # missing --age and --calibration

    def test_export_missing_args(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["export", "vizdata"])
        assert result.exit_code != 0  # missing --calibration and --output
