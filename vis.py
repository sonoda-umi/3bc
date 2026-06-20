"""Visualize experiment results using parallel coordinates plots.

This script reads CSV files generated under an experiment directory, filters by
solver and mode, rescales objective values, and writes one figure per selected
combination of mode, decision-space dimension, and objective dimension.

Usage example:
    python vis.py \
      --experiment-dir stats_output/2026-06-10T15:11:11+00:00 \
      --generation 200 \
      --output-dir charts \
      --show
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import polars as pl
import plotly.express as px
import plotly.graph_objects as go

FILE_NAME_PATTERN = "dim%DIM%_objs%OBJ%_tree_%mode%.csv"
DEFAULT_SOLVERS = ["NSGAIII", "IBEA", "MOEAD"]
DEFAULT_SOLVER_LABELS = {
    "IBEA": "IBEA",
    "MOEAD": "MOEA/D",
    "NSGAIII": "NSGA-III",
    "NSGAII": "NSGA-II",
    "OMOPSO": "OMOPSO",
    "GDE3": "GDE3",
}
DEFAULT_DIMENSIONS = ["root", "node_1", "node_2", "node_3", "node_4"]
SUPPORTED_MODES = ["depth", "breadth"]
MIN_CLIP_VALUE = 1e-1

PATTERN = re.compile(
    FILE_NAME_PATTERN
    .replace(r"%DIM%", r"(?P<dim>\d+)")
    .replace(r"%OBJ%", r"(?P<obj>\d+)")
    .replace(r"%mode%", r"(?P<mode>\w+)")
)


def find_generations(experiment_dir: Path) -> list[int]:
    """Return sorted generation numbers found under an experiment directory."""
    generation_dirs = [p.name for p in experiment_dir.iterdir() if p.is_dir()]
    generations = [int(name.split("_", 1)[1]) for name in generation_dirs if name.startswith("gen_")]
    generations.sort()
    return generations


def find_pairs(experiment_dir: Path, generation: int) -> list[tuple[int, int, str]]:
    """Return available (decision dim, objective dim, mode) tuples for a generation."""
    generation_dir = experiment_dir / f"gen_{generation}"
    if not generation_dir.exists():
        raise FileNotFoundError(f"Generation directory not found: {generation_dir}")

    pairs: set[tuple[int, int, str]] = set()
    for entry in generation_dir.iterdir():
        if not entry.is_file():
            continue
        match = PATTERN.search(entry.name)
        if not match:
            raise ValueError(f"Unexpected file name: {entry.name}")
        pairs.add((
            int(match.group("dim")),
            int(match.group("obj")),
            match.group("mode"),
        ))
    return sorted(pairs)


def read_file(experiment_dir: Path, generation: int, codim: int, objective_dim: int, mode: str) -> pl.DataFrame:
    """Read a single CSV file and annotate its metadata."""
    file_path = (
        experiment_dir
        / f"gen_{generation}"
        / FILE_NAME_PATTERN.replace("%DIM%", str(codim)).replace("%OBJ%", str(objective_dim)).replace("%mode%", mode)
    )
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pl.read_csv(file_path)
    return df.with_columns(
        [
            pl.lit(str(experiment_dir)).alias("experiment"),
            pl.lit(generation).alias("generation"),
            pl.lit(codim).alias("codim"),
            pl.lit(objective_dim).alias("objective dim"),
            pl.lit(mode).alias("mode"),
        ]
    )


def load_experiment_data(
    experiment_dir: Path,
    generation: int,
    modes: Iterable[str] = SUPPORTED_MODES,
) -> pl.DataFrame:
    """Load all matching files for a generation in an experiment directory."""
    pairs = find_pairs(experiment_dir=experiment_dir, generation=generation)
    selected_pairs = [pair for pair in pairs if pair[2] in modes]
    if not selected_pairs:
        raise ValueError(f"No matching mode files found in generation {generation}")

    frames = [
        read_file(experiment_dir=experiment_dir, generation=generation, codim=codim, objective_dim=obj_dim, mode=mode)
        for codim, obj_dim, mode in selected_pairs
    ]
    return pl.concat(frames, how="vertical")


def normalize_objectives(df: pl.DataFrame, dimensions: list[str]) -> pl.DataFrame:
    """Rescale objective values to log10 space with a lower bound clip."""
    return df.with_columns(
        [pl.col(dim).cast(pl.Float64).clip(lower_bound=MIN_CLIP_VALUE).log10().alias(dim) for dim in dimensions]
    )


def build_color_settings(solvers: list[str]) -> dict[str, object]:
    """Build a Plotly continuous color mapping for solver categories."""
    colors = px.colors.qualitative.Safe
    assert len(solvers) <= len(colors), "Not enough distinct colors for solvers"

    color_scale = []
    for index, solver in enumerate(solvers):
        color_scale.append([index / len(solvers), colors[index]])
        color_scale.append([(index + 1) / len(solvers), colors[index]])

    return {
        "color_continuous_scale": color_scale,
        "color_continuous_midpoint": 0.5 / len(solvers),
        "range_color": [0, len(solvers) - 1],
    }


def compose_plot_combinations(co_sum: int, modes: Iterable[str] = SUPPORTED_MODES) -> list[tuple[str, int, int]]:
    """Return all (mode, codim, objective dim) combinations for the given sum."""
    combinations = []
    for mode in modes:
        for codim in range(2, co_sum - 1):
            combinations.append((mode, codim, co_sum - codim))
    return combinations


def write_parallel_coordinates_plot(
    df: pl.DataFrame,
    mode: str,
    codim: int,
    objective_dim: int,
    dimensions: list[str],
    solvers: list[str],
    solver_labels: dict[str, str],
    output_dir: Path,
    show_plot: bool = False,
) -> None:
    """Create and save a parallel coordinates plot for a single combination."""
    subset = (
        df.filter(pl.col("mode") == mode)
        .filter(pl.col("codim") == codim)
        .filter(pl.col("objective dim") == objective_dim)
    )
    if subset.is_empty():
        return

    solver_category = {solver: idx for idx, solver in enumerate(solvers)}
    subset = subset.with_columns(pl.col("solver").replace_strict(solver_category, return_dtype=pl.Int64).alias("category_num"))

    fig = px.parallel_coordinates(
        subset,
        dimensions=dimensions,
        color="category_num",
        title=f"{subset[0, 'experiment']} - Generation {subset[0, 'generation']} - {mode} - codim={codim} - objective_dim={objective_dim}",
        **build_color_settings(solvers=solvers),
    )

    max_value = subset.select(dimensions).to_numpy().max()
    fig.update_traces(
        dimensions=[{"range": [-1, max_value]} for _ in dimensions]
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_layout(
        xaxis=dict(visible=False, showgrid=False, showticklabels=False),
        yaxis=dict(visible=False, showgrid=False, showticklabels=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    for solver, label in solver_labels.items():
        if solver not in solver_category:
            continue
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=px.colors.qualitative.Safe[solver_category[solver]], size=10),
                name=label,
                showlegend=True,
                legendgroup=f"solver_{solver}",
            )
        )
    output_sub_dir = output_dir / f"domain{codim+objective_dim-1}"
    output_sub_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_sub_dir / f"{mode}_codim{codim}_objective_dim{objective_dim}.png"
    fig.write_image(str(output_file), width=1818, height=450, scale=2)
    if show_plot:
        fig.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate parallel coordinates plots for experiment data.")
    parser.add_argument("--experiment-dir", type=Path, required=True, help="Path to the experiment directory containing gen_<N> subfolders.")
    parser.add_argument("--generation", type=int, help="Specific generation number to visualize. Defaults to the latest generation.")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"), help="Directory to save generated plot images.")
    parser.add_argument("--co-sum", type=int, default=5, help="Sum of codim and objective dim used to compose plot combinations.")
    parser.add_argument("--show", action="store_true", help="Display each plot after creation.")
    parser.add_argument("--solver", action="append", default=DEFAULT_SOLVERS, help="Solver name to include. Can be passed multiple times.")
    parser.add_argument("--mode", action="append", default=SUPPORTED_MODES, help="Mode to include (depth or breadth). Can be passed multiple times.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    generation = args.generation
    if generation is None:
        generations = find_generations(experiment_dir)
        if not generations:
            raise ValueError(f"No generation directories found in {experiment_dir}")
        generation = generations[-1]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_experiment_data(experiment_dir=experiment_dir, generation=generation, modes=args.mode)
    df = df.filter(pl.col("solver").is_in(args.solver))
    df = normalize_objectives(df=df, dimensions=DEFAULT_DIMENSIONS)

    plot_combinations = compose_plot_combinations(co_sum=args.co_sum, modes=args.mode)
    for mode, codim, objective_dim in plot_combinations:
        write_parallel_coordinates_plot(
            df=df,
            mode=mode,
            codim=codim,
            objective_dim=objective_dim,
            dimensions=DEFAULT_DIMENSIONS,
            solvers=args.solver,
            solver_labels=DEFAULT_SOLVER_LABELS,
            output_dir=output_dir,
            show_plot=args.show,
        )

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
