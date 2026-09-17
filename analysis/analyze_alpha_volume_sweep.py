from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GROUP_KEYS = ("synthetic_fraction", "activity_multiplier")
METRICS = (
    "expected_activation_share",
    "synthetic_activation_share",
    "synthetic_production_share",
    "synthetic_exposure_share",
    "amplification_ratio",
)


def load_runs(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))

    numeric_fields = {
        "synthetic_fraction", "activity_multiplier", "simulation_seed", "n_synthetic_agents",
        "expected_activation_share", "synthetic_activation_share", "synthetic_production_share",
        "synthetic_exposure_share", "amplification_ratio", "synthetic_posts", "human_posts",
        "synthetic_exposures", "human_exposures", "total_experimental_posts", "total_exposures",
    }

    for row in rows:
        for field in numeric_fields:
            row[field] = float(row[field])

        row["exposure_minus_production"] = row["synthetic_exposure_share"] - row["synthetic_production_share"]

    return rows


def finite(values):
    return np.asarray([x for x in values if math.isfinite(x)], dtype=float)


def summarize(values):
    x = finite(values)

    if len(x) == 0:
        return {"mean": math.nan, "std": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)

    for row in rows:
        groups[(row["synthetic_fraction"], row["activity_multiplier"])].append(row)

    output = []

    for (fraction, multiplier), group in sorted(groups.items()):
        result = {
            "synthetic_fraction": fraction,
            "activity_multiplier": multiplier,
            "n_runs": len(group),
        }

        for metric in METRICS + ("exposure_minus_production",):
            stats = summarize([row[metric] for row in group])

            for stat_name, value in stats.items():
                result[f"{metric}_{stat_name}"] = value

        amplification = finite([row["amplification_ratio"] for row in group])
        delta = finite([row["exposure_minus_production"] for row in group])

        result["fraction_amplification_gt_1"] = float(np.mean(amplification > 1.0)) if len(amplification) else math.nan
        result["fraction_exposure_gt_production"] = float(np.mean(delta > 0.0)) if len(delta) else math.nan

        output.append(result)

    return output


def save_summary(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: list[dict]):
    print("\n=== Alpha volume sweep aggregate ===")
    print("fraction  multiplier  production  exposure  mean delta  median delta  delta std  A>1")

    for row in summary:
        fraction = row["synthetic_fraction"]
        multiplier = row["activity_multiplier"]

        if fraction == 0:
            print(f"{fraction:7.1%}  {multiplier:10.1f}  {row['synthetic_activation_share_mean']:10.3f}  "
                  f"{row['synthetic_production_share_mean']:10.3f}  {row['synthetic_exposure_share_mean']:8.3f}  "
                  f"{'n/a':>13}  {row['exposure_minus_production_mean']:+8.3f}  {'n/a':>4}")
            continue

        print(
            f"{fraction:7.1%}  "
            f"{multiplier:10.1f}  "
            f"{row['synthetic_production_share_mean']:10.3f}  "
            f"{row['synthetic_exposure_share_mean']:8.3f}  "
            f"{row['exposure_minus_production_mean']:+10.3f}  "
            f"{row['exposure_minus_production_median']:+12.3f}  "
            f"{row['exposure_minus_production_std']:9.3f}  "
            f"{row['fraction_amplification_gt_1']:4.0%}")

def plot_exposure_delta(summary: list[dict], output_dir: Path):
    rows = [row for row in summary if row["synthetic_fraction"] > 0]

    fig, ax = plt.subplots(figsize=(7, 5))

    for fraction in sorted({row["synthetic_fraction"] for row in rows}):
        subset = sorted(
            (row for row in rows if row["synthetic_fraction"] == fraction),
            key=lambda row: row["activity_multiplier"],
        )

        x = [row["activity_multiplier"] for row in subset]
        y = [row["exposure_minus_production_mean"] for row in subset]
        yerr = [row["exposure_minus_production_std"] for row in subset]

        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=f"{fraction:.0%} synthetic")

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10], labels=["1×", "2×", "5×", "10×"])
    ax.set_xlabel("Synthetic activity multiplier")
    ax.set_ylabel("Exposure share − production share")
    ax.set_title("Disproportionate synthetic-content exposure")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "exposure_minus_production.png", dpi=200)
    plt.close(fig)

def plot_production_vs_exposure(summary: list[dict], output_dir: Path):
    rows = [row for row in summary if row["synthetic_fraction"] > 0]

    fig, ax = plt.subplots(figsize=(7, 6))

    for fraction in sorted({row["synthetic_fraction"] for row in rows}):
        subset = sorted((row for row in rows if row["synthetic_fraction"] == fraction), key=lambda x: x["activity_multiplier"])
        production = [row["synthetic_production_share_mean"] for row in subset]
        exposure = [row["synthetic_exposure_share_mean"] for row in subset]

        ax.plot(production, exposure, marker="o", label=f"{fraction:.0%} synthetic")

        for row, x, y in zip(subset, production, exposure):
            ax.annotate(f"{row['activity_multiplier']:g}x", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)

    limits = ax.get_xlim()
    upper = max(limits[1], ax.get_ylim()[1])
    ax.plot([0, upper], [0, upper], linestyle="--", linewidth=1, label="Exposure = production")

    ax.set_xlabel("Mean synthetic production share")
    ax.set_ylabel("Mean synthetic exposure share")
    ax.set_title("Synthetic production versus exposure")
    ax.legend()
    fig.tight_layout()

    path = output_dir / "production_vs_exposure.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_amplification(summary: list[dict], output_dir: Path):
    rows = [row for row in summary if row["synthetic_fraction"] > 0]

    fig, ax = plt.subplots(figsize=(7, 5))

    for fraction in sorted({row["synthetic_fraction"] for row in rows}):
        subset = sorted((row for row in rows if row["synthetic_fraction"] == fraction), key=lambda x: x["activity_multiplier"])
        x = [row["activity_multiplier"] for row in subset]
        y = [row["amplification_ratio_mean"] for row in subset]
        yerr = [row["amplification_ratio_std"] for row in subset]

        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=f"{fraction:.0%} synthetic")

    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10], labels=["1×", "2×", "5×", "10×"])
    ax.set_xlabel("Synthetic activity multiplier")
    ax.set_ylabel("Exposure / production")
    ax.set_title("Synthetic-content exposure amplification")
    ax.legend()
    fig.tight_layout()

    path = output_dir / "amplification_ratio.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_activation_validation(summary: list[dict], output_dir: Path):
    rows = [row for row in summary if row["synthetic_fraction"] > 0]

    expected = np.asarray([row["expected_activation_share_mean"] for row in rows])
    realized = np.asarray([row["synthetic_activation_share_mean"] for row in rows])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(expected, realized)

    upper = max(expected.max(), realized.max()) * 1.05
    ax.plot([0, upper], [0, upper], linestyle="--", linewidth=1)

    ax.set_xlabel("Expected synthetic activation share")
    ax.set_ylabel("Realized synthetic activation share")
    ax.set_title("Weighted-activation validation")
    fig.tight_layout()

    path = output_dir / "activation_validation.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/alpha_volume_sweep/runs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/alpha_volume_sweep"))
    parser.add_argument("--summary", type=Path, default=Path("results/alpha_volume_sweep/summary.csv"))
    args = parser.parse_args()

    rows = load_runs(args.input)
    summary = aggregate(rows)

    save_summary(summary, args.summary)
    print_summary(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_production_vs_exposure(summary, args.output_dir)
    plot_exposure_delta(summary, args.output_dir)
    plot_amplification(summary, args.output_dir)
    plot_activation_validation(summary, args.output_dir)

    print(f"\nSaved summary: {args.summary}")
    print(f"Saved figures: {args.output_dir}")


if __name__ == "__main__":
    main()