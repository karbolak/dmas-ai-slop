from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import os
import time

import numpy as np

import doces
import yaml


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def generate_replicate_seeds(master_seed: int, n_replicates: int) -> list[dict]:
    rng = random.Random(master_seed)
    replicates = []

    for replicate in range(n_replicates):
        replicates.append({
            "replicate": replicate,
            "network_seed": rng.randrange(0, 2**31),
            "opinion_seed": rng.randrange(0, 2**31),
            "producer_assignment_seed": rng.randrange(0, 2**31),
            "simulation_seed": rng.randrange(0, 2**31),
        })

    return replicates


def generate_directed_er_network(n_agents: int, mean_in_degree: float, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    p = mean_in_degree / (n_agents - 1)
    edges = []

    for source in range(n_agents):
        for target in range(n_agents):
            if source != target and rng.random() < p:
                edges.append((source, target))

    return edges


def generate_initial_opinions(n_agents: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(n_agents)]


def generate_producer_order(n_agents: int, seed: int) -> list[int]:
    users = list(range(n_agents))
    rng = random.Random(seed)
    rng.shuffle(users)
    return users


def get_synthetic_users(producer_order: list[int], fraction: float) -> set[int]:
    n_synthetic = int(round(len(producer_order) * fraction))
    return set(producer_order[:n_synthetic])


def expected_activation_share(n_agents: int, n_synthetic: int, multiplier: float) -> float:
    if n_synthetic == 0:
        return 0.0

    n_human = n_agents - n_synthetic
    return (n_synthetic * multiplier) / (n_human + n_synthetic * multiplier)

def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def gini_nonnegative(values) -> float:
    x = np.asarray(values, dtype=float)

    if x.size == 0:
        return math.nan

    total = float(x.sum())

    if total == 0:
        return 0.0

    x = np.sort(x)
    n = len(x)
    weights = np.arange(1, n + 1)

    return float((2.0 * np.sum(weights * x)) / (n * total) - (n + 1) / n)


def top_fraction_share(values, fraction: float) -> float:
    x = np.asarray(values, dtype=float)

    if x.size == 0:
        return math.nan

    total = float(x.sum())

    if total == 0:
        return 0.0

    k = max(1, int(math.ceil(len(x) * fraction)))
    return float(np.sort(x)[-k:].sum() / total)


def distribution_metrics(values, prefix: str) -> dict:
    x = np.asarray(values, dtype=float)

    if x.size == 0:
        return {
            f"{prefix}_zero_fraction": math.nan,
            f"{prefix}_p50": math.nan,
            f"{prefix}_p90": math.nan,
            f"{prefix}_p99": math.nan,
            f"{prefix}_max": math.nan,
            f"{prefix}_gini": math.nan,
            f"{prefix}_top_1pct_share": math.nan,
            f"{prefix}_top_10pct_share": math.nan,
        }

    return {
        f"{prefix}_zero_fraction": float(np.mean(x == 0)),
        f"{prefix}_p50": float(np.percentile(x, 50)),
        f"{prefix}_p90": float(np.percentile(x, 90)),
        f"{prefix}_p99": float(np.percentile(x, 99)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_gini": gini_nonnegative(x),
        f"{prefix}_top_1pct_share": top_fraction_share(x, 0.01),
        f"{prefix}_top_10pct_share": top_fraction_share(x, 0.10),
    }


def initial_network_metrics(edges, synthetic_users: set[int], n_agents: int) -> dict:
    indegree = np.zeros(n_agents, dtype=int)
    outdegree = np.zeros(n_agents, dtype=int)

    ss = sh = hs = hh = 0

    for source, target in edges:
        outdegree[source] += 1
        indegree[target] += 1

        source_synthetic = source in synthetic_users
        target_synthetic = target in synthetic_users

        if source_synthetic and target_synthetic:
            ss += 1
        elif source_synthetic:
            sh += 1
        elif target_synthetic:
            hs += 1
        else:
            hh += 1

    synthetic = np.asarray(sorted(synthetic_users), dtype=int)
    human = np.asarray([i for i in range(n_agents) if i not in synthetic_users], dtype=int)

    def group_mean(array, indices):
        return float(np.mean(array[indices])) if len(indices) else math.nan

    return {
        "synthetic_initial_mean_indegree": group_mean(indegree, synthetic),
        "synthetic_initial_mean_outdegree": group_mean(outdegree, synthetic),
        "human_initial_mean_indegree": group_mean(indegree, human),
        "human_initial_mean_outdegree": group_mean(outdegree, human),
        "initial_edges_synthetic_to_synthetic": ss,
        "initial_edges_synthetic_to_human": sh,
        "initial_edges_human_to_synthetic": hs,
        "initial_edges_human_to_human": hh,
    }

def analyse_run(model, synthetic_users: set[int], exclude_background_posts: bool = True) -> dict:
    origins = model.post_origin_user_ids
    received_counts = model.post_received_counts
    activation_counts = model.activation_counts

    synthetic_receipts_per_post = []
    human_receipts_per_post = []

    synthetic_posts = human_posts = 0
    synthetic_exposures = human_exposures = 0

    for origin, received in zip(origins, received_counts):
        origin = int(origin)
        received = int(received)

        if origin < 0:
            if exclude_background_posts:
                continue

            human_posts += 1
            human_exposures += received
            human_receipts_per_post.append(received)
            continue

        if origin in synthetic_users:
            synthetic_posts += 1
            synthetic_exposures += received
            synthetic_receipts_per_post.append(received)
        else:
            human_posts += 1
            human_exposures += received
            human_receipts_per_post.append(received)

    total_posts = synthetic_posts + human_posts
    total_exposures = synthetic_exposures + human_exposures

    production_share = safe_ratio(synthetic_posts, total_posts)
    exposure_share = safe_ratio(synthetic_exposures, total_exposures)

    delta = (
        exposure_share - production_share
        if math.isfinite(production_share) and math.isfinite(exposure_share)
        else math.nan
    )

    amplification_ratio = (
        exposure_share / production_share
        if production_share > 0 and math.isfinite(exposure_share)
        else math.nan
    )

    total_activations = sum(activation_counts)
    synthetic_activations = sum(activation_counts[user] for user in synthetic_users)
    human_activations = total_activations - synthetic_activations

    activation_share = safe_ratio(synthetic_activations, total_activations)

    metrics = {
        "synthetic_activation_share": activation_share,
        "synthetic_production_share": production_share,
        "synthetic_exposure_share": exposure_share,
        "exposure_minus_production": delta,
        "amplification_ratio": amplification_ratio,

        "synthetic_posts": synthetic_posts,
        "human_posts": human_posts,
        "synthetic_exposures": synthetic_exposures,
        "human_exposures": human_exposures,
        "total_experimental_posts": total_posts,
        "total_exposures": total_exposures,

        "synthetic_activations": synthetic_activations,
        "human_activations": human_activations,
        "total_activations": total_activations,

        "synthetic_exposures_per_post": safe_ratio(synthetic_exposures, synthetic_posts),
        "human_exposures_per_post": safe_ratio(human_exposures, human_posts),

        "synthetic_exposures_per_activation": safe_ratio(synthetic_exposures, synthetic_activations),
        "human_exposures_per_activation": safe_ratio(human_exposures, human_activations),

        "synthetic_posts_per_activation": safe_ratio(synthetic_posts, synthetic_activations),
        "human_posts_per_activation": safe_ratio(human_posts, human_activations),

        "synthetic_production_majority": float(production_share > 0.5) if math.isfinite(production_share) else math.nan,
        "synthetic_exposure_majority": float(exposure_share > 0.5) if math.isfinite(exposure_share) else math.nan,

        "synthetic_production_margin_50": production_share - 0.5 if math.isfinite(production_share) else math.nan,
        "synthetic_exposure_margin_50": exposure_share - 0.5 if math.isfinite(exposure_share) else math.nan,
    }

    metrics.update(distribution_metrics(synthetic_receipts_per_post, "synthetic_post_receipts"))
    metrics.update(distribution_metrics(human_receipts_per_post, "human_post_receipts"))

    return metrics


def run_condition(cfg: dict, edges: list[tuple[int, int]], initial_opinions: list[float], producer_order: list[int],
                  replicate_info: dict, synthetic_fraction: float, activity_multiplier: float) -> dict:
    n_agents = cfg["network"]["n_agents"]
    synthetic_users = get_synthetic_users(producer_order, synthetic_fraction)
    activity_weights = [activity_multiplier if user in synthetic_users else 1.0 for user in range(n_agents)]

    model = doces.Opinion_dynamics(vertex_count=n_agents, edges=edges, directed=cfg["network"]["directed"], verbose=False)
    model.set_activity_weights(activity_weights)
    
    started = time.perf_counter()
    network_metrics = initial_network_metrics(edges, synthetic_users, n_agents)

    sim_cfg = cfg["simulation"]
    model.simulate_dynamics(
        number_of_iterations=sim_cfg["iterations"], phi=sim_cfg["phi"], mu=sim_cfg["mu"],
        posting_filter=sim_cfg["posting_filter"], receiving_filter=sim_cfg["receiving_filter"],
        b=list(initial_opinions), feed_size=sim_cfg["feed_size"], rewire=sim_cfg["rewire"],
        min_opinion=-1.0, max_opinion=1.0, delta=sim_cfg["delta"], verbose=False,
        rand_seed=replicate_info["simulation_seed"],
    )

    metrics = analyse_run(model, synthetic_users, cfg["analysis"].get("exclude_background_posts", True))
    n_synthetic = len(synthetic_users)

    result = {
        "replicate": replicate_info["replicate"],
        "network_seed": replicate_info["network_seed"],
        "opinion_seed": replicate_info["opinion_seed"],
        "producer_assignment_seed": replicate_info["producer_assignment_seed"],
        "simulation_seed": replicate_info["simulation_seed"],
        "edge_count": len(edges),
        "realized_mean_in_degree": len(edges) / n_agents,
        "synthetic_fraction": synthetic_fraction,
        "activity_multiplier": activity_multiplier,
        "n_synthetic_agents": n_synthetic,
        "expected_activation_share": expected_activation_share(n_agents, n_synthetic, activity_multiplier),
    }

    result.update(metrics)
    result["elapsed_seconds"] = time.perf_counter() - started
    result.update(network_metrics)
    return result


def save_results(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_seed_manifest(replicates: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(replicates[0].keys()))
        writer.writeheader()
        writer.writerows(replicates)


def print_result(result: dict) -> None:
    print(
        f"rep={result['replicate']:02d} "
        f"fraction={result['synthetic_fraction']:.0%} "
        f"activity={result['activity_multiplier']:g}x "
        f"act={result['synthetic_activation_share']:.3f} "
        f"prod={result['synthetic_production_share']:.3f} "
        f"exp={result['synthetic_exposure_share']:.3f} "
        f"delta={result['exposure_minus_production']:+.3f}"
    )
    
def load_completed_keys(path: Path) -> set[tuple[int, float, float]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()

    with path.open() as f:
        rows = csv.DictReader(f)

        return {
            (
                int(float(row["replicate"])),
                float(row["synthetic_fraction"]),
                float(row["activity_multiplier"]),
            )
            for row in rows
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Run replicate 0 at 5% synthetic and 5x activity.")
    parser.add_argument("--replicates", type=int, default=None, help="Override the number of replicates in the config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_agents = cfg["network"]["n_agents"]
    n_replicates = args.replicates if args.replicates is not None else cfg["seeds"]["replicates"]
    replicates = generate_replicate_seeds(cfg["seeds"]["master"], n_replicates)

    output_dir = Path("results") / cfg["experiment"]["name"]
    save_seed_manifest(replicates, output_dir / "seed_manifest.csv")

    if args.smoke:
        replicate_info = replicates[0]
        edges = generate_directed_er_network(n_agents, cfg["network"]["mean_in_degree"], replicate_info["network_seed"])
        initial_opinions = generate_initial_opinions(n_agents, replicate_info["opinion_seed"])
        producer_order = generate_producer_order(n_agents, replicate_info["producer_assignment_seed"])

        result = run_condition(cfg, edges, initial_opinions, producer_order, replicate_info, 0.05, 5.0)
        print("\n=== Independent-world smoke test ===")
        print(f"Network seed:   {replicate_info['network_seed']}")
        print(f"Opinion seed:   {replicate_info['opinion_seed']}")
        print(f"Producer seed:  {replicate_info['producer_assignment_seed']}")
        print(f"Simulation seed:{replicate_info['simulation_seed']}")
        print(f"Edges:          {len(edges)}")
        print_result(result)
        return

    fractions = cfg["synthetic"]["fractions"]
    multipliers = cfg["synthetic"]["activity_multipliers"]

    output_path = output_dir / "runs.csv"
    completed = load_completed_keys(output_path)

    existing_header = None

    if output_path.exists() and output_path.stat().st_size > 0:
        with output_path.open() as f:
            existing_header = next(csv.reader(f))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = output_path.open("a", newline="")
    writer = None

    try:
        for replicate_info in replicates:
            replicate = replicate_info["replicate"]
            print(f"\n=== Replicate {replicate + 1}/{n_replicates} ===")

            edges = generate_directed_er_network(
                n_agents,
                cfg["network"]["mean_in_degree"],
                replicate_info["network_seed"],
            )

            initial_opinions = generate_initial_opinions(
                n_agents,
                replicate_info["opinion_seed"],
            )

            producer_order = generate_producer_order(
                n_agents,
                replicate_info["producer_assignment_seed"],
            )

            for fraction in fractions:
                condition_multipliers = [1.0] if fraction == 0.0 else multipliers

                for multiplier in condition_multipliers:
                    key = (replicate, float(fraction), float(multiplier))

                    if key in completed:
                        continue

                    result = run_condition(
                        cfg,
                        edges,
                        initial_opinions,
                        producer_order,
                        replicate_info,
                        synthetic_fraction=fraction,
                        activity_multiplier=multiplier,
                    )

                    if writer is None:
                        fieldnames = list(result.keys())

                        if existing_header is not None and existing_header != fieldnames:
                            raise RuntimeError(
                                "Existing runs.csv schema differs from the current output schema. "
                                "Use a fresh experiment name."
                            )

                        writer = csv.DictWriter(output_file, fieldnames=fieldnames)

                        if existing_header is None:
                            writer.writeheader()

                    writer.writerow(result)
                    output_file.flush()

                    completed.add(key)
                    if (replicate + 1) % 100 == 0 or replicate == 0 or replicate + 1 == n_replicates:
                        print(
                            f"Completed world {replicate + 1}/{n_replicates} "
                            f"({len(completed)} condition-runs saved)"
    )

    finally:
        output_file.close()

    condition_count = 1 + (len(fractions) - 1) * len(multipliers)

    print(
        f"\nTarget complete: {n_replicates} worlds × "
        f"{condition_count} conditions = {n_replicates * condition_count} runs"
    )
    print(f"Results: {output_path}")
    print(f"Seeds:   {output_dir / 'seed_manifest.csv'}")


if __name__ == "__main__":
    main()