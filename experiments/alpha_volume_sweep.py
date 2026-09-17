from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import doces
import yaml


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


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
    synthetic_weight = n_synthetic * multiplier
    return synthetic_weight / (n_human + synthetic_weight)


def analyse_run(model, synthetic_users: set[int]) -> dict:
    origins = model.post_origin_user_ids
    received_counts = model.post_received_counts
    activation_counts = model.activation_counts

    synthetic_posts = 0
    human_posts = 0
    synthetic_exposures = 0
    human_exposures = 0

    for origin, received in zip(origins, received_counts):
        if origin < 0:
            continue

        if origin in synthetic_users:
            synthetic_posts += 1
            synthetic_exposures += received
        else:
            human_posts += 1
            human_exposures += received

    total_posts = synthetic_posts + human_posts
    total_exposures = synthetic_exposures + human_exposures

    production_share = synthetic_posts / total_posts if total_posts > 0 else math.nan
    exposure_share = synthetic_exposures / total_exposures if total_exposures > 0 else math.nan

    if production_share > 0 and math.isfinite(exposure_share):
        amplification_ratio = exposure_share / production_share
    else:
        amplification_ratio = math.nan

    total_activations = sum(activation_counts)
    synthetic_activations = sum(activation_counts[user] for user in synthetic_users)
    activation_share = synthetic_activations / total_activations if total_activations > 0 else math.nan

    return {
        "synthetic_activation_share": activation_share,
        "synthetic_production_share": production_share,
        "synthetic_exposure_share": exposure_share,
        "amplification_ratio": amplification_ratio,
        "synthetic_posts": synthetic_posts,
        "human_posts": human_posts,
        "synthetic_exposures": synthetic_exposures,
        "human_exposures": human_exposures,
        "total_experimental_posts": total_posts,
        "total_exposures": total_exposures,
    }


def run_condition(
    cfg: dict,
    edges: list[tuple[int, int]],
    initial_opinions: list[float],
    producer_order: list[int],
    synthetic_fraction: float,
    activity_multiplier: float,
    simulation_seed: int,
) -> dict:
    n_agents = cfg["network"]["n_agents"]
    synthetic_users = get_synthetic_users(producer_order, synthetic_fraction)

    activity_weights = [
        activity_multiplier if user in synthetic_users else 1.0
        for user in range(n_agents)
    ]

    model = doces.Opinion_dynamics(
        vertex_count=n_agents,
        edges=edges,
        directed=cfg["network"]["directed"],
        verbose=False,
    )

    model.set_activity_weights(activity_weights)

    sim_cfg = cfg["simulation"]

    model.simulate_dynamics(
        number_of_iterations=sim_cfg["iterations"],
        phi=sim_cfg["phi"],
        mu=sim_cfg["mu"],
        posting_filter=sim_cfg["posting_filter"],
        receiving_filter=sim_cfg["receiving_filter"],
        b=list(initial_opinions),
        feed_size=sim_cfg["feed_size"],
        rewire=sim_cfg["rewire"],
        min_opinion=-1.0,
        max_opinion=1.0,
        delta=sim_cfg["delta"],
        verbose=False,
        rand_seed=simulation_seed,
    )

    metrics = analyse_run(model, synthetic_users)
    n_synthetic = len(synthetic_users)

    result = {
        "synthetic_fraction": synthetic_fraction,
        "activity_multiplier": activity_multiplier,
        "simulation_seed": simulation_seed,
        "n_synthetic_agents": n_synthetic,
        "expected_activation_share": expected_activation_share(n_agents, n_synthetic, activity_multiplier),
    }

    result.update(metrics)
    return result


def print_result(result: dict) -> None:
    print("\n=== Alpha volume condition ===")
    print(f"Synthetic fraction:        {result['synthetic_fraction']:.1%}")
    print(f"Activity multiplier:       {result['activity_multiplier']:.1f}x")
    print(f"Simulation seed:           {result['simulation_seed']}")
    print(f"Synthetic agents:          {result['n_synthetic_agents']}")
    print(f"Expected activation share: {result['expected_activation_share']:.4f}")
    print(f"Realized activation share: {result['synthetic_activation_share']:.4f}")
    print(f"Production share:          {result['synthetic_production_share']:.4f}")
    print(f"Exposure share:            {result['synthetic_exposure_share']:.4f}")
    print(f"Amplification ratio:       {result['amplification_ratio']:.4f}")
    print(f"Synthetic posts:           {result['synthetic_posts']}")
    print(f"Human posts:               {result['human_posts']}")
    print(f"Synthetic exposures:       {result['synthetic_exposures']}")
    print(f"Human exposures:           {result['human_exposures']}")


def save_results(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Run only one 5% synthetic, 5x activity condition.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_agents = cfg["network"]["n_agents"]

    print("Generating fixed network...")
    edges = generate_directed_er_network(
        n_agents=n_agents,
        mean_in_degree=cfg["network"]["mean_in_degree"],
        seed=cfg["seeds"]["network"],
    )
    print(f"Edges: {len(edges)}")

    initial_opinions = generate_initial_opinions(n_agents, cfg["seeds"]["opinions"])
    producer_order = generate_producer_order(n_agents, cfg["seeds"]["producer_assignment"])

    rows = []

    if args.smoke:
        result = run_condition(
            cfg=cfg,
            edges=edges,
            initial_opinions=initial_opinions,
            producer_order=producer_order,
            synthetic_fraction=0.05,
            activity_multiplier=5.0,
            simulation_seed=0,
        )

        print_result(result)
        return

    fractions = cfg["synthetic"]["fractions"]
    multipliers = cfg["synthetic"]["activity_multipliers"]
    seed_start = cfg["seeds"]["simulation"]["start"]
    seed_stop = cfg["seeds"]["simulation"]["stop"]

    for fraction in fractions:
        condition_multipliers = [1.0] if fraction == 0.0 else multipliers

        for multiplier in condition_multipliers:
            for seed in range(seed_start, seed_stop):
                print(f"Running fraction={fraction:.2f}, multiplier={multiplier:g}, seed={seed}")

                result = run_condition(
                    cfg=cfg,
                    edges=edges,
                    initial_opinions=initial_opinions,
                    producer_order=producer_order,
                    synthetic_fraction=fraction,
                    activity_multiplier=multiplier,
                    simulation_seed=seed,
                )

                rows.append(result)
                print_result(result)

    output_path = Path("results") / cfg["experiment"]["name"] / "runs.csv"
    save_results(rows, output_path)

    print(f"\nSaved {len(rows)} runs to {output_path}")


if __name__ == "__main__":
    main()