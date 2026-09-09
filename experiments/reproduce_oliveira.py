"""
Representative reproduction of the Oliveira et al. Figure 2 baseline.

The experiment compares two runs with identical network, initial
opinions, model parameters, and simulation seed while changing only
the innovation probability:

    mu = 1.0
    mu = 0.1

The network and initial opinions correspond to the fixed state used
in the 50-seed validation sweep. Simulation seed 6 was selected after
that sweep as a representative Figure-2-like realization and is used
for visualization only.

Usage:

    python experiments/reproduce_oliveira.py \
        --config configs/oliveira_fig2.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml

import doces


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_directed_er_network(
    n_agents: int,
    mean_in_degree: float,
    seed: int,
) -> list[tuple[int, int]]:
    """
    Generate a directed Erdos-Renyi network.

    For a directed graph without self-loops:
        expected mean in-degree = p * (N - 1)

    Therefore:
        p = z / (N - 1)
    """
    if n_agents < 2:
        raise ValueError("n_agents must be at least 2.")

    p = mean_in_degree / (n_agents - 1)

    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"Invalid edge probability {p:.6f}; check N and mean degree."
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


def make_initial_opinions(n_agents: int, seed: int) -> list[float]:
    """Uniform initial opinions in [-1, 1]."""
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(n_agents)]


def run_condition(
    *,
    n_agents: int,
    edges: list[tuple[int, int]],
    initial_opinions: list[float],
    iterations: int,
    phi: float,
    mu: float,
    posting_filter: int,
    receiving_filter: int,
    feed_size: int,
    rewire: bool,
    delta: float,
    simulation_seed: int,
) -> dict[str, Any]:
    """Run one DOCES condition on a fresh model instance."""

    model = doces.Opinion_dynamics(
        n_agents,
        edges,
        True,
    )

    result = model.simulate_dynamics(
        iterations,
        phi,
        mu,
        posting_filter,
        receiving_filter,
        b=list(initial_opinions),
        feed_size=feed_size,
        rewire=rewire,
        cascade_stats_output_file=None,
        min_opinion=-1,
        max_opinion=1,
        delta=delta,
        verbose=False,
        rand_seed=simulation_seed,
    )

    cascade_stats = model.get_cascade_stats_dict()

    return {
        "opinions": np.asarray(result["b"], dtype=float),
        "edges": list(result["edges"]),
        "cascade_stats": cascade_stats,
    }

def bimodality_coefficient(values: np.ndarray) -> float:
    """
    Bimodality coefficient used by Oliveira et al.

    A distribution is typically considered bimodal when:
        BC > 5/9
    """
    n = len(values)

    if n < 4:
        return float("nan")

    g = float(skew(values, bias=False))
    k = float(kurtosis(values, fisher=True, bias=False))

    correction = (
        3.0 * (n - 1) ** 2
        / ((n - 2) * (n - 3))
    )

    return (g**2 + 1.0) / (k + correction)


def cascade_summary(cascade_stats: dict[str, Any]) -> dict[str, Any]:
    sizes = np.asarray(cascade_stats.get("cascade_size", []), dtype=float)

    if sizes.size == 0:
        return {
            "n_cascades": 0,
            "mean_cascade_size": None,
            "median_cascade_size": None,
            "max_cascade_size": None,
        }

    return {
        "n_cascades": int(sizes.size),
        "mean_cascade_size": float(sizes.mean()),
        "median_cascade_size": float(np.median(sizes)),
        "max_cascade_size": float(sizes.max()),
    }


def save_edges(path: Path, edges: list[tuple[int, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target"])
        writer.writerows(edges)


def plot_opinion_comparison(
    results: dict[str, dict[str, Any]],
    bins: int,
    output_path: Path,
) -> None:
    """Save one figure with overlaid final opinion histograms."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, result in results.items():
        ax.hist(
            result["opinions"],
            bins=bins,
            alpha=0.45,
            density=True,
            label=name,
        )

    ax.set_xlabel("Final opinion")
    ax.set_ylabel("Density")
    ax.set_title("Oliveira Figure 2 representative: final opinion distributions")    
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_network_sample(
    edges: list[tuple[int, int]],
    opinions: np.ndarray,
    output_path: Path,
    simulation_seed: int,
    max_nodes: int = 250,
) -> None:
    """
    Save a lightweight visual diagnostic of the final network.

    For readability, only a deterministic subset of nodes is plotted
    when N is large.
    """
    g = nx.DiGraph()
    g.add_edges_from(edges)

    nodes = sorted(g.nodes())
    if len(nodes) > max_nodes:
        rng = random.Random(simulation_seed)
        nodes = sorted(rng.sample(nodes, max_nodes))
        g = g.subgraph(nodes).copy()

    pos = nx.spring_layout(g, seed=simulation_seed)

    node_values = [opinions[node] for node in g.nodes()]

    fig, ax = plt.subplots(figsize=(7, 7))
    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        alpha=0.12,
        arrows=False,
        width=0.5,
    )
    nodes_artist = nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_size=18,
        node_color=node_values,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    fig.colorbar(nodes_artist, ax=ax, label="Opinion")
    ax.set_title("Sample of final network")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

def opinion_sign_assortativity(
    edges: list[tuple[int, int]],
    opinions: np.ndarray,
) -> float:
    g = nx.DiGraph()
    g.add_edges_from(edges)

    for node in g.nodes:
        if opinions[node] > 0:
            g.nodes[node]["sign"] = "positive"
        elif opinions[node] < 0:
            g.nodes[node]["sign"] = "negative"
        else:
            g.nodes[node]["sign"] = "zero"

    return float(
        nx.attribute_assortativity_coefficient(
            g,
            "sign",
        )
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML experiment configuration.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    
    if not bool(cfg["network"].get("directed", True)):
        raise ValueError(
            "The Oliveira Figure 2 baseline requires a directed network."
        )

    exp_name = cfg["experiment"]["name"]
    network_seed = int(cfg["seeds"]["network"])
    opinions_seed = int(cfg["seeds"]["opinions"])
    simulation_seed = int(cfg["seeds"]["simulation"])
    n_agents = int(cfg["network"]["n_agents"])
    mean_in_degree = float(cfg["network"]["mean_in_degree"])

    results_dir = Path("results") / exp_name
    figures_dir = Path("figures") / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=== Oliveira Figure 2 representative reproduction ===")
    print(f"Experiment:       {exp_name}")
    print(f"Agents:           {n_agents}")
    print(f"Mean in-degree:   {mean_in_degree}")
    print(f"Iterations:       {cfg['simulation']['iterations']}")
    print(f"Network seed:      {network_seed}")
    print(f"Opinion seed:      {opinions_seed}")
    print(f"Simulation seed:   {simulation_seed}")
    print()

    edges = make_directed_er_network(
        n_agents=n_agents,
        mean_in_degree=mean_in_degree,
        seed=network_seed,
    )

    initial_opinions = make_initial_opinions(
        n_agents=n_agents,
        seed=opinions_seed,
    )

    realised_mean_in_degree = len(edges) / n_agents

    print(f"Generated edges:          {len(edges)}")
    print(f"Realised mean in-degree:  {realised_mean_in_degree:.3f}")
    print(f"Initial opinion mean:     {np.mean(initial_opinions):.4f}")
    print()

    all_results: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}

    sim = cfg["simulation"]

    for condition in cfg["conditions"]:
        name = str(condition["name"])
        mu = float(condition["mu"])

        print(f"--- Running {name}: mu={mu} ---")

        result = run_condition(
            n_agents=n_agents,
            edges=edges,
            initial_opinions=initial_opinions,
            iterations=int(sim["iterations"]),
            phi=float(sim["phi"]),
            mu=mu,
            posting_filter=int(sim["posting_filter"]),
            receiving_filter=int(sim["receiving_filter"]),
            feed_size=int(sim["feed_size"]),
            rewire=bool(sim["rewire"]),
            delta=float(sim["delta"]),
            simulation_seed=simulation_seed,
        )

        all_results[name] = result

        opinions = result["opinions"]
        bc = bimodality_coefficient(opinions)
        summary = {
            "mu": mu,
            "mean_opinion": float(opinions.mean()),
            "std_opinion": float(opinions.std()),
            "min_opinion": float(opinions.min()),
            "max_opinion": float(opinions.max()),
            "bimodality_coefficient": bc,
            "positive_fraction": float(np.mean(opinions > 0)),
            "negative_fraction": float(np.mean(opinions < 0)),
            "is_bimodal": bc > 5 / 9,
            "opinion_sign_assortativity":
            opinion_sign_assortativity(
                result["edges"],
                opinions,
            ),
            "final_edge_count": len(result["edges"]),
            **cascade_summary(result["cascade_stats"]),
        }
        summaries[name] = summary

        np.savetxt(
            results_dir / f"{name}_opinions.csv",
            opinions,
            delimiter=",",
            header="opinion",
            comments="",
        )

        if cfg["analysis"].get("save_final_edges", True):
            save_edges(
                results_dir / f"{name}_edges.csv",
                result["edges"],
            )

        plot_network_sample(
            result["edges"],
            opinions,
            figures_dir / f"{name}_network_sample.png",
            simulation_seed=simulation_seed,
        )

        print(json.dumps(summary, indent=2))
        print()

    plot_opinion_comparison(
        all_results,
        bins=int(cfg["analysis"]["opinion_histogram_bins"]),
        output_path=figures_dir / "opinion_comparison.png",
    )

    run_metadata = {
        "config": cfg,
        "generated_network": {
            "edge_count": len(edges),
            "realised_mean_in_degree": realised_mean_in_degree,
        },
        "summaries": summaries,
        "replication_status": (
            "validated qualitative Oliveira innovation baseline; "
            "representative seed selected from 50-seed fixed-state sweep"
        ),
    }

    with (results_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    print("=== Completed ===")
    print(f"Results: {results_dir}")
    print(f"Figures: {figures_dir}")
    print()
    print(
    "NOTE: This run uses the published Figure 2 parameterization. "
    "Simulation seed 6 was selected after the 50-seed validation sweep "
    "as a representative Figure-2-like realization. It is used for "
    "visualization only; aggregate validation is based on the full seed sweep."
)


if __name__ == "__main__":
    main()
