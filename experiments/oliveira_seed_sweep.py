"""
Fixed-state paired simulation seed sweep for the Oliveira et al.
innovation-probability baseline.

A single network and initial opinion vector are generated once.
For each simulation seed, both mu conditions start from exactly
the same network and initial opinions.

Only the stochastic DOCES trajectory and mu condition vary.

Usage:

    python experiments/oliveira_seed_sweep.py \
        --config configs/oliveira_fig2.yaml \
        --seeds 50

Outputs:

    results/oliveira_fixed_state_seed_sweep/
        runs.csv
        summary.csv
        paired_differences.csv
        metadata.json

    figures/oliveira_fixed_state_seed_sweep/
        bc_by_mu.png
        assortativity_by_mu.png
        paired_bc.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml
from scipy.stats import kurtosis, skew

import doces


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------
# Network + initial state
# ---------------------------------------------------------------------


def make_directed_er_network(
    n_agents: int,
    mean_in_degree: float,
    seed: int,
) -> list[tuple[int, int]]:
    """
    Generate a directed Erdos-Renyi network.

    For a directed graph without self-loops:

        E[in-degree] = p * (N - 1)

    therefore:

        p = z / (N - 1)
    """
    if n_agents < 2:
        raise ValueError("n_agents must be >= 2.")

    p = mean_in_degree / (n_agents - 1)

    if not 0 <= p <= 1:
        raise ValueError(
            f"Invalid edge probability {p:.6f}. "
            "Check n_agents and mean_in_degree."
        )

    rng = random.Random(seed)

    edges: list[tuple[int, int]] = []

    for source in range(n_agents):
        for target in range(n_agents):
            if source == target:
                continue
            if rng.random() < p:
                edges.append((source, target))

    return edges


def make_initial_opinions(
    n_agents: int,
    seed: int,
) -> list[float]:
    """
    Generate initial opinions uniformly in [-1, 1].
    """
    rng = random.Random(seed)

    return [rng.uniform(-1.0, 1.0) for _ in range(n_agents)]


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def bimodality_coefficient(values: np.ndarray) -> float:
    """
    Bimodality coefficient used by Oliveira et al.

    BC = (g^2 + 1) /
         (k + 3(n-1)^2 / ((n-2)(n-3)))

    where:
        g = sample skewness
        k = excess kurtosis

    BC > 5/9 is typically treated as bimodal.
    """
    n = len(values)

    if n < 4:
        return float("nan")

    g = float(skew(values, bias=False))

    k = float(kurtosis(values, fisher=True, bias=False))

    correction = (3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))

    denominator = k + correction

    if denominator == 0:
        return float("nan")

    return (g**2 + 1.0) / denominator


def opinion_sign_assortativity(edges: list[tuple[int, int]], opinions: np.ndarray) -> float:
    """
    Network assortativity based on final opinion sign.

    Rough interpretation:

        ~0  -> no sign-based preference
        >0  -> like-minded users connect preferentially
        ~1  -> very strong segregation
    """
    if not edges:
        return float("nan")

    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    for node in range(len(opinions)):
        graph.add_node(node)

        opinion = opinions[node]

        if opinion > 0:
            sign = "positive"
        elif opinion < 0:
            sign = "negative"
        else:
            sign = "zero"

        graph.nodes[node]["opinion_sign"] = sign

    value = nx.attribute_assortativity_coefficient(graph, "opinion_sign")

    return float(value)


def cascade_summary(cascade_stats: dict[str, Any]) -> dict[str, Any]:
    """
    Return basic statistics for cascade sizes.
    """
    sizes = np.asarray(cascade_stats.get("cascade_size", []), dtype=float)

    if sizes.size == 0:
        return {
            "n_cascades": 0,
            "mean_cascade_size": float("nan"),
            "median_cascade_size": float("nan"),
            "max_cascade_size": float("nan"),
        }

    return {"n_cascades": int(sizes.size), "mean_cascade_size": float(sizes.mean()),
        "median_cascade_size": float(np.median(sizes)),
        "max_cascade_size": float(sizes.max())}


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------


def run_condition(*, n_agents: int, edges: list[tuple[int, int]], initial_opinions: list[float], iterations: int, phi: float, mu: float,
                posting_filter: int, receiving_filter: int, feed_size: int, rewire: bool, delta: float, simulation_seed: int
                ) -> dict[str, Any]:
    """
    Run one DOCES condition from a fixed supplied initial state.
    """
    model = doces.Opinion_dynamics(n_agents, edges, True)

    result = model.simulate_dynamics(iterations, phi, mu, posting_filter, receiving_filter, b=list(initial_opinions), feed_size=feed_size,
                                    rewire=rewire, cascade_stats_output_file=None, min_opinion=-1, max_opinion=1, delta=delta, verbose=False, rand_seed=simulation_seed)

    return {"opinions": np.asarray(result["b"], dtype=float), "edges": list(result["edges"]), "cascade_stats": model.get_cascade_stats_dict()}

# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    Write a list of dictionaries to CSV.
    """
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))

        writer.writeheader()
        writer.writerows(rows)

# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def finite(values: list[float]) -> np.ndarray:
    """
    Return finite values only.
    """
    arr = np.asarray(values, dtype=float,)

    return arr[np.isfinite(arr)]


def aggregate_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate all trajectories belonging to one mu condition.
    """
    bcs = finite([row["bimodality_coefficient"] for row in rows])

    assortativity = finite([row["opinion_sign_assortativity"] for row in rows])

    opinion_std = finite([row["std_opinion"] for row in rows])

    max_cascade = finite([row["max_cascade_size"] for row in rows])

    bimodal_values = [bool(row["is_bimodal"]) for row in rows]

    return {"condition": rows[0]["condition"], "mu": rows[0]["mu"], "n_runs": len(rows),
        "mean_bc": float(np.mean(bcs)),
        "median_bc": float(np.median(bcs)),
        "std_bc": float(np.std(bcs)),
        "bimodal_runs": int(sum(bimodal_values)),
        "bimodal_fraction": float(np.mean(bimodal_values)),
        "mean_opinion_std": float(np.mean(opinion_std)),
        "mean_assortativity": float(np.mean(assortativity)),
        "median_assortativity": float(np.median(assortativity)),
        "median_max_cascade_size": float(np.median(max_cascade)),
        "mean_max_cascade_size": float(np.mean(max_cascade))}


def build_paired_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Create one comparison row per simulation seed.

    The Figure 2 baseline compares:
        mu = 1.0
        mu = 0.1

    Each pair uses the same:
        network
        initial opinions
        simulation seed
    """
    by_seed: dict[int, dict[float, dict[str, Any]]] = {}

    for row in rows:
        seed = int(row["seed"])
        mu = float(row["mu"])
        by_seed.setdefault(seed, {})
        by_seed[seed][mu] = row

    output: list[dict[str, Any]] = []

    for seed, conditions in sorted(by_seed.items()):
        if (1.0 not in conditions or 0.1 not in conditions):
            continue

        mu_1 = conditions[1.0]
        mu_01 = conditions[0.1]

        output.append({"seed": seed,
                "delta_bc_mu01_minus_mu1": (mu_01["bimodality_coefficient"] - mu_1["bimodality_coefficient"]),
                "delta_assortativity_mu01_minus_mu1":(mu_01["opinion_sign_assortativity"] - mu_1["opinion_sign_assortativity"]),
                "delta_std_opinion_mu01_minus_mu1": (mu_01["std_opinion"] - mu_1["std_opinion"]),
                "delta_max_cascade_mu01_minus_mu1":(mu_01["max_cascade_size"] - mu_1["max_cascade_size"]),
                "mu1_bimodal": mu_1["is_bimodal"],
                "mu01_bimodal": mu_01["is_bimodal"]})

    return output

# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------


def plot_metric_by_mu(rows: list[dict[str, Any]], metric: str, ylabel: str, output_path: Path) -> None:
    """
    Boxplot of one metric across mu conditions.
    """
    mu_values = sorted({float(row["mu"]) for row in rows}, reverse=True)
    data = [[float(row[metric]) for row in rows if (float(row["mu"]) == mu and math.isfinite(float(row[metric])))] for mu in mu_values]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, tick_labels=[f"μ = {mu:g}" for mu in mu_values])
    ax.set_ylabel(ylabel)
    ax.set_title(f"Fixed-state seed sweep: {ylabel}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_paired_bc(rows: list[dict[str, Any]], output_path: Path) -> None:
    """
    Plot each paired seed as a line from mu=1.0 to mu=0.1.
    """
    by_seed: dict[int, dict[float, float]] = {}

    for row in rows:
        seed = int(row["seed"])
        mu = float(row["mu"])
        bc = float(row["bimodality_coefficient"])
        by_seed.setdefault(seed, {})
        by_seed[seed][mu] = bc

    fig, ax = plt.subplots(figsize=(7, 5))

    for seed in sorted(by_seed):
        pair = by_seed[seed]

        if 1.0 not in pair or 0.1 not in pair:
            continue

        ax.plot([1.0, 0.1], [pair[1.0], pair[0.1]], marker="o", alpha=0.35)

    ax.axhline(5 / 9, linestyle="--", label="BC = 5/9")
    ax.set_xticks([1.0, 0.1], ["μ = 1.0", "μ = 0.1"])
    ax.set_ylabel("Bimodality coefficient")
    ax.set_title("Paired change in bimodality by simulation seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    FIXED_NETWORK_SEED = 42
    FIXED_OPINIONS_SEED = 43

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to Oliveira Figure 2 YAML config.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=50,
        help="Number of paired simulation seeds to run.",
    )

    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="First simulation seed in the sweep.",
    )

    args = parser.parse_args()

    if args.seeds <= 0:
        raise ValueError(
            "--seeds must be > 0."
        )

    cfg = load_config(
        args.config
    )

    n_agents = int(
        cfg["network"][
            "n_agents"
        ]
    )

    mean_in_degree = float(
        cfg["network"][
            "mean_in_degree"
        ]
    )

    directed = bool(
        cfg["network"].get(
            "directed",
            True,
        )
    )

    if not directed:
        raise ValueError(
            "Oliveira baseline expects "
            "a directed network."
        )

    sim = cfg[
        "simulation"
    ]

    conditions = cfg[
        "conditions"
    ]

    # This script specifically reproduces the
    # Figure 2 comparison.
    mu_values = {
        float(
            condition["mu"]
        )
        for condition in conditions
    }

    required_mu_values = {
        1.0,
        0.1,
    }

    if not required_mu_values.issubset(
        mu_values
    ):
        raise ValueError(
            "Fixed-state Figure 2 sweep "
            "requires mu=1.0 and mu=0.1 "
            "conditions."
        )

    experiment_name = (
        "oliveira_fixed_state_seed_sweep"
    )

    results_dir = (
        Path("results")
        / experiment_name
    )

    figures_dir = (
        Path("figures")
        / experiment_name
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Generate the initial state ONCE.
    #
    # Every stochastic trajectory starts from exactly the same
    # network and exactly the same opinion distribution.
    # ---------------------------------------------------------

    network_seed = (
        FIXED_NETWORK_SEED
    )

    opinions_seed = (
        FIXED_OPINIONS_SEED
    )

    edges = make_directed_er_network(
        n_agents=n_agents,
        mean_in_degree=mean_in_degree,
        seed=network_seed,
    )

    initial_opinions = (
        make_initial_opinions(
            n_agents=n_agents,
            seed=opinions_seed,
        )
    )

    realised_mean_degree = (
        len(edges)
        / n_agents
    )

    initial_mean_opinion = float(
        np.mean(
            initial_opinions
        )
    )

    print(
        "=== Oliveira fixed-state paired seed sweep ==="
    )

    print(
        f"Simulation seeds: "
        f"{args.start_seed}.."
        f"{args.start_seed + args.seeds - 1}"
    )

    print(
        f"Network seed:     "
        f"{network_seed}"
    )

    print(
        f"Opinions seed:    "
        f"{opinions_seed}"
    )

    print(
        f"Agents:           "
        f"{n_agents}"
    )

    print(
        f"Edges:            "
        f"{len(edges)}"
    )

    print(
        f"Mean in-degree:   "
        f"{realised_mean_degree:.3f}"
    )

    print(
        f"Initial mean b:   "
        f"{initial_mean_opinion:.4f}"
    )

    print(
        f"Iterations:       "
        f"{sim['iterations']}"
    )

    print(
        "Conditions:       "
        + ", ".join(
            f"{condition['name']} "
            f"(mu={condition['mu']})"
            for condition in conditions
        )
    )

    print()

    all_rows: list[
        dict[str, Any]
    ] = []

    # ---------------------------------------------------------
    # Simulation-seed sweep.
    #
    # Network + initial opinions stay fixed.
    # Only DOCES randomness changes between trajectories.
    # ---------------------------------------------------------

    for sweep_index, seed in enumerate(
        range(
            args.start_seed,
            args.start_seed
            + args.seeds,
        ),
        start=1,
    ):
        print(
            f"=== Simulation seed {seed} "
            f"({sweep_index}/{args.seeds}) ==="
        )

        for condition in conditions:
            condition_name = str(
                condition[
                    "name"
                ]
            )

            mu = float(
                condition[
                    "mu"
                ]
            )

            print(
                f"  -> {condition_name}: "
                f"mu={mu}"
            )

            result = run_condition(
                n_agents=n_agents,
                edges=edges,
                initial_opinions=initial_opinions,
                iterations=int(
                    sim[
                        "iterations"
                    ]
                ),
                phi=float(
                    sim[
                        "phi"
                    ]
                ),
                mu=mu,
                posting_filter=int(
                    sim[
                        "posting_filter"
                    ]
                ),
                receiving_filter=int(
                    sim[
                        "receiving_filter"
                    ]
                ),
                feed_size=int(
                    sim[
                        "feed_size"
                    ]
                ),
                rewire=bool(
                    sim[
                        "rewire"
                    ]
                ),
                delta=float(
                    sim[
                        "delta"
                    ]
                ),
                simulation_seed=seed,
            )

            opinions = result[
                "opinions"
            ]

            bc = (
                bimodality_coefficient(
                    opinions
                )
            )

            assortativity = (
                opinion_sign_assortativity(
                    result[
                        "edges"
                    ],
                    opinions,
                )
            )

            cascade = (
                cascade_summary(
                    result[
                        "cascade_stats"
                    ]
                )
            )

            row = {
                "seed":
                    seed,

                "network_seed":
                    network_seed,

                "opinions_seed":
                    opinions_seed,

                "condition":
                    condition_name,

                "mu":
                    mu,

                "n_agents":
                    n_agents,

                "edge_count":
                    len(edges),

                "realised_mean_in_degree":
                    realised_mean_degree,

                "initial_mean_opinion":
                    initial_mean_opinion,

                "mean_opinion":
                    float(
                        opinions.mean()
                    ),

                "std_opinion":
                    float(
                        opinions.std()
                    ),

                "min_opinion":
                    float(
                        opinions.min()
                    ),

                "max_opinion":
                    float(
                        opinions.max()
                    ),

                "positive_fraction":
                    float(
                        np.mean(
                            opinions > 0
                        )
                    ),

                "negative_fraction":
                    float(
                        np.mean(
                            opinions < 0
                        )
                    ),

                "bimodality_coefficient":
                    bc,

                "is_bimodal":
                    bool(
                        bc > 5 / 9
                    ),

                "opinion_sign_assortativity":
                    assortativity,

                "final_edge_count":
                    len(
                        result[
                            "edges"
                        ]
                    ),

                **cascade,
            }

            all_rows.append(
                row
            )

            print(
                f"     BC={bc:.3f}, "
                f"bimodal="
                f"{row['is_bimodal']}, "
                f"assort="
                f"{assortativity:.3f}, "
                f"max cascade="
                f"{row['max_cascade_size']:.0f}"
            )

        print()

    # ---------------------------------------------------------
    # Save individual trajectories.
    # ---------------------------------------------------------

    write_csv(
        results_dir
        / "runs.csv",
        all_rows,
    )

    # ---------------------------------------------------------
    # Aggregate results by condition.
    # ---------------------------------------------------------

    summaries: list[
        dict[str, Any]
    ] = []

    for condition in conditions:
        name = str(
            condition[
                "name"
            ]
        )

        condition_rows = [
            row
            for row in all_rows
            if row[
                "condition"
            ]
            == name
        ]

        summaries.append(
            aggregate_condition(
                condition_rows
            )
        )

    write_csv(
        results_dir
        / "summary.csv",
        summaries,
    )

    # ---------------------------------------------------------
    # Paired differences.
    # ---------------------------------------------------------

    paired = (
        build_paired_differences(
            all_rows
        )
    )

    write_csv(
        results_dir
        / "paired_differences.csv",
        paired,
    )

    # ---------------------------------------------------------
    # Figures.
    # ---------------------------------------------------------

    plot_metric_by_mu(
        rows=all_rows,
        metric=(
            "bimodality_coefficient"
        ),
        ylabel=(
            "Bimodality coefficient"
        ),
        output_path=(
            figures_dir
            / "bc_by_mu.png"
        ),
    )

    plot_metric_by_mu(
        rows=all_rows,
        metric=(
            "opinion_sign_assortativity"
        ),
        ylabel=(
            "Opinion-sign assortativity"
        ),
        output_path=(
            figures_dir
            / "assortativity_by_mu.png"
        ),
    )

    plot_paired_bc(
        rows=all_rows,
        output_path=(
            figures_dir
            / "paired_bc.png"
        ),
    )

    # ---------------------------------------------------------
    # Metadata.
    # ---------------------------------------------------------

    metadata = {
        "experiment":
            experiment_name,

        "config":
            cfg,

        "fixed_initial_state": {
            "network_seed":
                network_seed,

            "opinions_seed":
                opinions_seed,

            "edge_count":
                len(edges),

            "realised_mean_in_degree":
                realised_mean_degree,

            "initial_mean_opinion":
                initial_mean_opinion,
        },

        "simulation_seed_sweep": {
            "start_seed":
                args.start_seed,

            "n_seeds":
                args.seeds,
        },

        "bimodality_threshold":
            5 / 9,

        "summaries":
            summaries,
    }

    with (
        results_dir
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # Console summary.
    # ---------------------------------------------------------

    print()
    print(
        "=== Fixed-state seed sweep complete ==="
    )

    for summary in summaries:
        print()

        print(
            f"{summary['condition']} "
            f"(mu={summary['mu']}):"
        )

        print(
            f"  mean BC:             "
            f"{summary['mean_bc']:.4f}"
        )

        print(
            f"  median BC:           "
            f"{summary['median_bc']:.4f}"
        )

        print(
            f"  bimodal runs:        "
            f"{summary['bimodal_runs']}/"
            f"{summary['n_runs']} "
            f"("
            f"{summary['bimodal_fraction']:.1%}"
            f")"
        )

        print(
            f"  mean assortativity:  "
            f"{summary['mean_assortativity']:.4f}"
        )

        print(
            f"  mean opinion std:    "
            f"{summary['mean_opinion_std']:.4f}"
        )

        print(
            f"  median max cascade:  "
            f"{summary['median_max_cascade_size']:.1f}"
        )

    if paired:
        delta_bc = np.asarray(
            [
                row[
                    "delta_bc_mu01_minus_mu1"
                ]
                for row in paired
            ],
            dtype=float,
        )

        delta_assort = np.asarray(
            [
                row[
                    "delta_assortativity_mu01_minus_mu1"
                ]
                for row in paired
            ],
            dtype=float,
        )

        delta_std = np.asarray(
            [
                row[
                    "delta_std_opinion_mu01_minus_mu1"
                ]
                for row in paired
            ],
            dtype=float,
        )

        delta_cascade = np.asarray(
            [
                row[
                    "delta_max_cascade_mu01_minus_mu1"
                ]
                for row in paired
            ],
            dtype=float,
        )

        print()

        print(
            "Paired differences "
            "(mu=0.1 minus mu=1.0):"
        )

        print(
            f"  mean delta BC:        "
            f"{np.mean(delta_bc):.4f}"
        )

        print(
            f"  median delta BC:      "
            f"{np.median(delta_bc):.4f}"
        )

        print(
            f"  seeds with delta BC > 0: "
            f"{np.mean(delta_bc > 0):.1%}"
        )

        print(
            f"  mean delta assort.:   "
            f"{np.mean(delta_assort):.4f}"
        )

        print(
            f"  mean delta std:       "
            f"{np.mean(delta_std):.4f}"
        )

        print(
            f"  median delta max cascade: "
            f"{np.median(delta_cascade):.1f}"
        )

    print()

    print(
        f"Results: {results_dir}"
    )

    print(
        f"Figures: {figures_dir}"
    )


if __name__ == "__main__":
    main()