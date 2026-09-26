from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator
import numpy as np
from scipy.stats import t


METRICS = (
    "expected_activation_share",
    "synthetic_activation_share",
    "synthetic_production_share",
    "synthetic_exposure_share",
    "exposure_minus_production",
    "amplification_ratio",

    "synthetic_exposures_per_post",
    "human_exposures_per_post",
    "synthetic_exposures_per_activation",
    "human_exposures_per_activation",
    "synthetic_posts_per_activation",
    "human_posts_per_activation",

    "synthetic_post_receipts_zero_fraction",
    "human_post_receipts_zero_fraction",
    "synthetic_post_receipts_p50",
    "synthetic_post_receipts_p90",
    "synthetic_post_receipts_p99",
    "synthetic_post_receipts_max",
    "human_post_receipts_p50",
    "human_post_receipts_p90",
    "human_post_receipts_p99",
    "human_post_receipts_max",

    "synthetic_post_receipts_gini",
    "human_post_receipts_gini",
    "synthetic_post_receipts_top_1pct_share",
    "human_post_receipts_top_1pct_share",
    "synthetic_post_receipts_top_10pct_share",
    "human_post_receipts_top_10pct_share",

    "synthetic_production_majority",
    "synthetic_exposure_majority",
    "synthetic_production_margin_50",
    "synthetic_exposure_margin_50",

    "synthetic_initial_mean_indegree",
    "synthetic_initial_mean_outdegree",
    "human_initial_mean_indegree",
    "human_initial_mean_outdegree",

    "elapsed_seconds",
)


def load_runs(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))

    numeric_fields = {
        "replicate", "network_seed", "opinion_seed", "producer_assignment_seed", "simulation_seed",
        "edge_count", "realized_mean_in_degree", "synthetic_fraction", "activity_multiplier",
        "n_synthetic_agents", "expected_activation_share", "synthetic_activation_share",
        "synthetic_production_share", "synthetic_exposure_share", "exposure_minus_production",
        "amplification_ratio", "synthetic_posts", "human_posts", "synthetic_exposures",
        "human_exposures", "total_experimental_posts", "total_exposures",
    }

    for row in rows:
        for field, value in row.items():
            if value == "":
                continue

            try:
                row[field] = float(value)
            except ValueError:
                pass

        row["exposure_minus_production"] = (
            row["synthetic_exposure_share"] - row["synthetic_production_share"]
        )

    return rows
        
        


def finite(values) -> np.ndarray:
    return np.asarray([x for x in values if math.isfinite(x)], dtype=float)


def summarize(values) -> dict:
    x = finite(values)

    if len(x) == 0:
        return {
            "n": 0, "mean": math.nan, "std": math.nan, "median": math.nan,
            "min": math.nan, "max": math.nan, "ci95_low": math.nan,
            "ci95_high": math.nan, "ci95_half_width": math.nan,
        }

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

    if len(x) > 1:
        half_width = float(t.ppf(0.975, df=len(x) - 1) * std / math.sqrt(len(x)))
    else:
        half_width = 0.0

    return {
        "n": len(x),
        "mean": mean,
        "std": std,
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "ci95_half_width": half_width,
    }


def validate_design(rows: list[dict]) -> None:
    conditions = defaultdict(set)

    for row in rows:
        key = (row["synthetic_fraction"], row["activity_multiplier"])
        conditions[key].add(int(row["replicate"]))

    replicate_ids = sorted({int(row["replicate"]) for row in rows})
    expected_replicates = set(replicate_ids)

    print("=== Design validation ===")
    print(f"Rows:                 {len(rows)}")
    print(f"Independent worlds:   {len(replicate_ids)}")
    print(f"Experimental conditions: {len(conditions)}")

    problems = []

    for condition, replicates in sorted(conditions.items()):
        if replicates != expected_replicates:
            problems.append((condition, len(replicates)))

    if problems:
        print("WARNING: Some conditions do not contain all replicate worlds:")
        for condition, n in problems:
            print(f"  {condition}: {n}/{len(expected_replicates)} worlds")
    else:
        print("✓ Every condition contains the same replicate worlds.")

    activation_errors = [
        abs(row["synthetic_activation_share"] - row["expected_activation_share"])
        for row in rows if row["synthetic_fraction"] > 0
    ]

    print(f"Mean |expected-realized activation|: {np.mean(activation_errors):.6f}")
    print(f"Max  |expected-realized activation|: {np.max(activation_errors):.6f}")


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)

    for row in rows:
        groups[(row["synthetic_fraction"], row["activity_multiplier"])].append(row)

    output = []

    for (fraction, multiplier), group in sorted(groups.items()):
        result = {
            "synthetic_fraction": fraction,
            "activity_multiplier": multiplier,
            "n_worlds": len({int(row["replicate"]) for row in group}),
        }

        for metric in METRICS:
            stats = summarize([row[metric] for row in group])

            for stat_name, value in stats.items():
                result[f"{metric}_{stat_name}"] = value

        delta = finite([row["exposure_minus_production"] for row in group])
        amplification = finite([row["amplification_ratio"] for row in group])

        result["fraction_exposure_gt_production"] = float(np.mean(delta > 0)) if len(delta) else math.nan
        result["fraction_amplification_gt_1"] = float(np.mean(amplification > 1)) if len(amplification) else math.nan

        output.append(result)

    return output


def paired_activity_effects(rows: list[dict]) -> list[dict]:
    by_condition = {}

    for row in rows:
        key = (int(row["replicate"]), row["synthetic_fraction"], row["activity_multiplier"])
        by_condition[key] = row

    fractions = sorted({row["synthetic_fraction"] for row in rows if row["synthetic_fraction"] > 0})
    multipliers = sorted({row["activity_multiplier"] for row in rows if row["activity_multiplier"] > 1})
    output = []

    for fraction in fractions:
        for multiplier in multipliers:
            paired = []

            for replicate in sorted({int(row["replicate"]) for row in rows}):
                baseline = by_condition.get((replicate, fraction, 1.0))
                treatment = by_condition.get((replicate, fraction, multiplier))

                if baseline is None or treatment is None:
                    continue

                paired.append({
                    "replicate": replicate,
                    "effect": treatment["exposure_minus_production"] - baseline["exposure_minus_production"],
                    "production_change": treatment["synthetic_production_share"] - baseline["synthetic_production_share"],
                    "exposure_change": treatment["synthetic_exposure_share"] - baseline["synthetic_exposure_share"],
                })

            effect_stats = summarize([x["effect"] for x in paired])
            production_stats = summarize([x["production_change"] for x in paired])
            exposure_stats = summarize([x["exposure_change"] for x in paired])

            result = {
                "synthetic_fraction": fraction,
                "activity_multiplier": multiplier,
                "baseline_multiplier": 1.0,
                "n_worlds": len(paired),
            }

            for name, stats in (
                ("paired_delta_effect", effect_stats),
                ("paired_production_change", production_stats),
                ("paired_exposure_change", exposure_stats),
            ):
                for stat_name, value in stats.items():
                    result[f"{name}_{stat_name}"] = value

            effects = finite([x["effect"] for x in paired])
            result["fraction_paired_effect_gt_0"] = float(np.mean(effects > 0)) if len(effects) else math.nan

            output.append(result)

    return output


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: list[dict]) -> None:
    print("\n=== Independent-world aggregate ===")
    print("fraction  activity   production   exposure   mean D     95% CI D          median D   D>0")

    for row in summary:
        fraction = row["synthetic_fraction"]
        multiplier = row["activity_multiplier"]

        if fraction == 0:
            print(
                f"{fraction:7.1%}  {multiplier:7.1f}x  "
                f"{row['synthetic_production_share_mean']:10.3f}  "
                f"{row['synthetic_exposure_share_mean']:8.3f}  "
                f"{row['exposure_minus_production_mean']:+8.3f}  "
                f"{'[control]':>18}  "
                f"{row['exposure_minus_production_median']:+9.3f}  {'n/a':>4}"
            )
            continue

        print(
            f"{fraction:7.1%}  {multiplier:7.1f}x  "
            f"{row['synthetic_production_share_mean']:10.3f}  "
            f"{row['synthetic_exposure_share_mean']:8.3f}  "
            f"{row['exposure_minus_production_mean']:+8.3f}  "
            f"[{row['exposure_minus_production_ci95_low']:+.3f}, "
            f"{row['exposure_minus_production_ci95_high']:+.3f}]  "
            f"{row['exposure_minus_production_median']:+9.3f}  "
            f"{row['fraction_exposure_gt_production']:4.0%}"
        )


def print_paired_effects(paired: list[dict]) -> None:
    print("\n=== Paired activity effects relative to 1x ===")
    print("fraction  activity   mean ΔD     95% CI ΔD          median ΔD   ΔD>0")

    for row in paired:
        print(
            f"{row['synthetic_fraction']:7.1%}  "
            f"{row['activity_multiplier']:7.1f}x  "
            f"{row['paired_delta_effect_mean']:+8.3f}  "
            f"[{row['paired_delta_effect_ci95_low']:+.3f}, "
            f"{row['paired_delta_effect_ci95_high']:+.3f}]  "
            f"{row['paired_delta_effect_median']:+10.3f}  "
            f"{row['fraction_paired_effect_gt_0']:5.0%}"
        )


def plot_exposure_delta(summary: list[dict], output_dir: Path) -> None:
    rows = [row for row in summary if row["synthetic_fraction"] > 0]
    fig, ax = plt.subplots(figsize=(7, 5))

    for fraction in sorted({row["synthetic_fraction"] for row in rows}):
        subset = sorted(
            (row for row in rows if row["synthetic_fraction"] == fraction),
            key=lambda row: row["activity_multiplier"],
        )

        x = np.asarray([row["activity_multiplier"] for row in subset])
        y = np.asarray([row["exposure_minus_production_mean"] for row in subset])
        low = np.asarray([row["exposure_minus_production_ci95_low"] for row in subset])
        high = np.asarray([row["exposure_minus_production_ci95_high"] for row in subset])

        line, = ax.plot(x, y, marker="o", label=f"{fraction:.0%} synthetic")
        ax.fill_between(x, low, high, alpha=0.12, color=line.get_color())

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    activity_levels = sorted({row["activity_multiplier"] for row in rows})
    ax.set_xticks(activity_levels, labels=[f"{x:g}×" for x in activity_levels])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Synthetic activity multiplier")
    ax.set_ylabel("Exposure share − production share")
    ax.set_title("Synthetic-content exposure relative to production")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "exposure_minus_production_shaded.png", dpi=200)
    plt.close(fig)


def plot_paired_activity_effects(paired: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    for fraction in sorted({row["synthetic_fraction"] for row in paired}):
        subset = sorted(
            (row for row in paired if row["synthetic_fraction"] == fraction),
            key=lambda row: row["activity_multiplier"],
        )

        x = np.asarray([row["activity_multiplier"] for row in subset])
        y = np.asarray([row["paired_delta_effect_mean"] for row in subset])
        low = np.asarray([row["paired_delta_effect_ci95_low"] for row in subset])
        high = np.asarray([row["paired_delta_effect_ci95_high"] for row in subset])

        line, = ax.plot(x, y, marker="o", label=f"{fraction:.0%} synthetic")
        ax.fill_between(x, low, high, alpha=0.12, color=line.get_color())

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    activity_levels = sorted({row["activity_multiplier"] for row in paired})
    ax.set_xticks(activity_levels, labels=[f"{x:g}×" for x in activity_levels])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Synthetic activity multiplier")
    ax.set_ylabel(r"Change in $D$ relative to 1×")
    ax.set_title("Paired effect of increased synthetic activity")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "paired_activity_effects_shaded.png", dpi=200)
    plt.close(fig)


def plot_production_vs_exposure(summary: list[dict], output_dir: Path) -> None:
    rows = [row for row in summary if row["synthetic_fraction"] > 0]
    fig, ax = plt.subplots(figsize=(7, 6))

    for fraction in sorted({row["synthetic_fraction"] for row in rows}):
        subset = sorted((row for row in rows if row["synthetic_fraction"] == fraction), key=lambda row: row["activity_multiplier"])
        production = [row["synthetic_production_share_mean"] for row in subset]
        exposure = [row["synthetic_exposure_share_mean"] for row in subset]

        ax.plot(production, exposure, marker="o", label=f"{fraction:.0%} synthetic")

        for row, x, y in zip(subset, production, exposure):
            ax.annotate(f"{row['activity_multiplier']:g}×", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)

    upper = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, upper], [0, upper], linestyle="--", linewidth=1, label="Exposure = production")
    ax.set_xlabel("Mean synthetic production share")
    ax.set_ylabel("Mean synthetic exposure share")
    ax.set_title("Synthetic production versus exposure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "production_vs_exposure.png", dpi=200)
    plt.close(fig)


def plot_activation_validation(summary: list[dict], output_dir: Path) -> None:
    rows = [row for row in summary if row["synthetic_fraction"] > 0]
    expected = np.asarray([row["expected_activation_share_mean"] for row in rows])
    realized = np.asarray([row["synthetic_activation_share_mean"] for row in rows])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(expected, realized)

    upper = max(expected.max(), realized.max()) * 1.05
    ax.plot([0, upper], [0, upper], linestyle="--", linewidth=1)
    ax.set_xlabel("Expected synthetic activation share")
    ax.set_ylabel("Realised synthetic activation share")
    ax.set_title("Weighted-activation validation")
    fig.tight_layout()
    fig.savefig(output_dir / "activation_validation.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/alpha_volume_sweep/runs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/alpha_volume_sweep"))
    parser.add_argument("--summary", type=Path, default=Path("results/alpha_volume_sweep/summary.csv"))
    parser.add_argument("--paired", type=Path, default=Path("results/alpha_volume_sweep/paired_activity_effects.csv"))
    args = parser.parse_args()

    rows = load_runs(args.input)
    validate_design(rows)

    summary = aggregate(rows)
    paired = paired_activity_effects(rows)

    save_csv(summary, args.summary)
    save_csv(paired, args.paired)

    print_summary(summary)
    print_paired_effects(paired)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_production_vs_exposure(summary, args.output_dir)
    plot_exposure_delta(summary, args.output_dir)
    plot_paired_activity_effects(paired, args.output_dir)
    plot_activation_validation(summary, args.output_dir)

    print(f"\nSaved summary:        {args.summary}")
    print(f"Saved paired effects: {args.paired}")
    print(f"Saved figures:        {args.output_dir}")


if __name__ == "__main__":
    main()