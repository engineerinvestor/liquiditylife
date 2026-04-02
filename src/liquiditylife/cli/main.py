"""Command-line interface for liquiditylife."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from liquiditylife import __version__


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def cli(*, verbose: bool) -> None:
    """liquiditylife: Life-cycle portfolio choice with liquidity risk."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


@cli.command("list-calibrations")
def list_calibrations_cmd() -> None:
    """List available calibration bundles."""
    from liquiditylife.calibrations.registry import list_calibrations

    for name in list_calibrations():
        click.echo(name)


@cli.command()
@click.option("--calibration", "-c", required=True, help="Calibration name.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output path.")
def solve(calibration: str, output: str | None) -> None:
    """Solve the lifecycle model for a calibration."""
    from liquiditylife.calibrations.registry import load_calibration
    from liquiditylife.io.cache import save_solution
    from liquiditylife.solve.solver import solve_model

    cal = load_calibration(calibration)
    click.echo(f"Solving {calibration}...")
    solution = solve_model(cal)
    out_path = save_solution(solution, Path(output) if output else None)
    click.echo(f"Saved to {out_path}")


@cli.command()
@click.option("--calibration", "-c", required=True, help="Calibration name.")
@click.option("--n", "n_households", type=int, default=10_000, help="Number of households.")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output CSV path.")
def simulate(calibration: str, n_households: int, seed: int, output: str | None) -> None:
    """Simulate household cohorts."""
    from liquiditylife.calibrations.registry import load_calibration
    from liquiditylife.solve.solver import solve_model
    from liquiditylife.sweep.surface import age_profile

    cal = load_calibration(calibration)
    click.echo(f"Solving {calibration}...")
    solution = solve_model(cal)
    click.echo(f"Simulating {n_households} households...")
    profile = age_profile(solution, n_households=n_households, seed=seed)

    if output:
        profile.to_csv(output, index=False)
        click.echo(f"Saved age profile to {output}")
    else:
        click.echo(profile.to_string(index=False))


@cli.command("sweep")
@click.argument("sweep_type", type=click.Choice(["policy-surface"]))
@click.option("--age", type=int, required=True, help="Age for surface.")
@click.option("--calibration", "-c", required=True, help="Calibration name.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output CSV path.")
def sweep_cmd(sweep_type: str, age: int, calibration: str, output: str | None) -> None:
    """Generate parameter sweeps."""
    from liquiditylife.calibrations.registry import load_calibration
    from liquiditylife.solve.solver import solve_model
    from liquiditylife.sweep.surface import policy_surface

    cal = load_calibration(calibration)
    click.echo(f"Solving {calibration}...")
    solution = solve_model(cal)
    click.echo(f"Computing policy surface at age {age}...")
    surface = policy_surface(solution, age=age)

    if output:
        surface.to_csv(output, index=False)
        click.echo(f"Saved surface to {output}")
    else:
        click.echo(surface.to_string(index=False))


@cli.command("export")
@click.argument("export_type", type=click.Choice(["vizdata"]))
@click.option("--calibration", "-c", required=True, help="Calibration name.")
@click.option(
    "--format", "fmt", type=click.Choice(["parquet", "json", "csv"]),
    default="csv", help="Output format.",
)
@click.option("--output", "-o", type=click.Path(), required=True, help="Output path.")
def export_cmd(export_type: str, calibration: str, fmt: str, output: str) -> None:
    """Export visualization data."""
    from liquiditylife.calibrations.registry import load_calibration
    from liquiditylife.solve.solver import solve_model
    from liquiditylife.vizdata.export import solution_to_dataframe, to_parquet

    cal = load_calibration(calibration)
    click.echo(f"Solving {calibration}...")
    solution = solve_model(cal)
    df = solution_to_dataframe(solution)

    out_path = Path(output)
    if fmt == "parquet":
        to_parquet(df, out_path)
    elif fmt == "json":
        out_path.write_text(df.to_json(orient="records"))
    else:
        df.to_csv(out_path, index=False)

    click.echo(f"Exported to {out_path}")
