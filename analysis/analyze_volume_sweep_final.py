from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
from scipy.stats import chi2, t


KEYS = ["replicate", "synthetic_fraction", "activity_multiplier"]

BASE_COLS = [
    "replicate",
    "network_seed",
    "opinion_seed",
    "producer_assignment_seed",
    "simulation_seed",
    "synthetic_fraction",
    "activity_multiplier",
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
    "synthetic_production_majority",
    "synthetic_exposure_majority",
    "synthetic_post_receipts_zero_fraction",
    "human_post_receipts_zero_fraction",
    "synthetic_post_receipts_p50",
    "human_post_receipts_p50",
    "synthetic_post_receipts_p90",
    "human_post_receipts_p90",
    "synthetic_post_receipts_p99",
    "human_post_receipts_p99",
    "synthetic_post_receipts_gini",
    "human_post_receipts_gini",
    "synthetic_post_receipts_top_1pct_share",
    "human_post_receipts_top_1pct_share",
    "synthetic_post_receipts_top_10pct_share",
    "human_post_receipts_top_10pct_share",
]

DERIVED_METRICS = [
    "exposure_efficiency_gap",
    "exposure_efficiency_ratio",
    "activation_efficiency_gap",
    "zero_receipt_gap",
    "p50_receipt_gap",
    "p90_receipt_gap",
    "p99_receipt_gap",
    "gini_gap",
    "top_1pct_share_gap",
    "top_10pct_share_gap",
]

SUMMARY_METRICS = [
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
    "human_post_receipts_p50",
    "synthetic_post_receipts_p90",
    "human_post_receipts_p90",
    "synthetic_post_receipts_p99",
    "human_post_receipts_p99",
    "synthetic_post_receipts_gini",
    "human_post_receipts_gini",
    "synthetic_post_receipts_top_1pct_share",
    "human_post_receipts_top_1pct_share",
    "synthetic_post_receipts_top_10pct_share",
    "human_post_receipts_top_10pct_share",
] + DERIVED_METRICS

DUPLICATE_CHECK_COLS = [
    "synthetic_activation_share",
    "synthetic_production_share",
    "synthetic_exposure_share",
    "exposure_minus_production",
    "synthetic_exposures_per_post",
    "human_exposures_per_post",
    "synthetic_post_receipts_zero_fraction",
    "human_post_receipts_zero_fraction",
]


# ---------- utilities ----------

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def finite_array(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return x[np.isfinite(x)]


def mean_ci(values, confidence: float = 0.95) -> dict:
    x = finite_array(values)
    n = len(x)
    if n == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "median": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "ci95_half_width": math.nan,
        }

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    half = float(t.ppf((1 + confidence) / 2, df=n - 1) * std / math.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "median": float(np.median(x)),
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "ci95_half_width": half,
    }


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    # 1.959963984540054 is the 97.5th standard-normal percentile.
    z = 1.959963984540054 if confidence == 0.95 else 1.959963984540054
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / denom
    return centre - half, centre + half


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    finite_mask = np.isfinite(p)
    idx = np.where(finite_mask)[0]
    if len(idx) == 0:
        return adjusted.tolist()

    ordered = idx[np.argsort(p[idx])]
    m = len(ordered)
    running = 0.0
    for rank, original_idx in enumerate(ordered):
        candidate = min(1.0, (m - rank) * p[original_idx])
        running = max(running, candidate)
        adjusted[original_idx] = running
    return adjusted.tolist()


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def activity_label(x: float) -> str:
    return f"{x:g}×"


# ---------- loading / validation ----------

def load_runs(path: Path) -> tuple[pd.DataFrame, dict]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in BASE_COLS if c not in header]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    df = pd.read_csv(path, usecols=BASE_COLS)
    raw_rows = len(df)

    key_counts = df.groupby(KEYS, sort=False).size()
    duplicate_keys = int((key_counts > 1).sum())
    duplicate_rows = int(raw_rows - len(key_counts))
    max_multiplicity = int(key_counts.max())

    inconsistent_duplicate_keys = 0
    if duplicate_keys:
        dup_mask = df.duplicated(KEYS, keep=False)
        core_unique = df.loc[dup_mask, KEYS + DUPLICATE_CHECK_COLS].drop_duplicates()
        inconsistent_duplicate_keys = int((core_unique.groupby(KEYS, sort=False).size() > 1).sum())

    # Deterministic rule: for each replicate-condition key, keep the first row in file order.
    df = df.drop_duplicates(KEYS, keep="first").copy()

    df["exposure_minus_production"] = (
        df["synthetic_exposure_share"] - df["synthetic_production_share"]
    )
    df["exposure_efficiency_gap"] = (
        df["synthetic_exposures_per_post"] - df["human_exposures_per_post"]
    )
    df["exposure_efficiency_ratio"] = (
        df["synthetic_exposures_per_post"] / df["human_exposures_per_post"]
    )
    df["activation_efficiency_gap"] = (
        df["synthetic_exposures_per_activation"] - df["human_exposures_per_activation"]
    )
    df["zero_receipt_gap"] = (
        df["synthetic_post_receipts_zero_fraction"] - df["human_post_receipts_zero_fraction"]
    )
    df["p50_receipt_gap"] = df["synthetic_post_receipts_p50"] - df["human_post_receipts_p50"]
    df["p90_receipt_gap"] = df["synthetic_post_receipts_p90"] - df["human_post_receipts_p90"]
    df["p99_receipt_gap"] = df["synthetic_post_receipts_p99"] - df["human_post_receipts_p99"]
    df["gini_gap"] = df["synthetic_post_receipts_gini"] - df["human_post_receipts_gini"]
    df["top_1pct_share_gap"] = (
        df["synthetic_post_receipts_top_1pct_share"] - df["human_post_receipts_top_1pct_share"]
    )
    df["top_10pct_share_gap"] = (
        df["synthetic_post_receipts_top_10pct_share"] - df["human_post_receipts_top_10pct_share"]
    )

    positive = df[df["synthetic_fraction"] > 0].copy()
    sign_d = np.sign(positive["exposure_minus_production"].to_numpy())
    sign_eff = np.sign(positive["exposure_efficiency_gap"].to_numpy())
    finite = np.isfinite(sign_d) & np.isfinite(sign_eff)
    sign_match = float(np.mean(sign_d[finite] == sign_eff[finite])) if finite.any() else math.nan

    validation = {
        "raw_rows": raw_rows,
        "unique_rows_after_dedup": len(df),
        "duplicate_keys": duplicate_keys,
        "duplicate_rows_removed": duplicate_rows,
        "max_key_multiplicity": max_multiplicity,
        "inconsistent_duplicate_keys_on_core_metrics": inconsistent_duplicate_keys,
        "unique_replicates": int(df["replicate"].nunique()),
        "unique_conditions": int(df[["synthetic_fraction", "activity_multiplier"]].drop_duplicates().shape[0]),
        "D_sign_matches_exposure_efficiency_gap_fraction": sign_match,
    }
    return df, validation


def validate_balanced_design(df: pd.DataFrame) -> tuple[list[float], list[float]]:
    positive = df[df["synthetic_fraction"] > 0]
    fractions = sorted(positive["synthetic_fraction"].unique().tolist())
    activities = sorted(positive["activity_multiplier"].unique().tolist())
    expected = len(fractions) * len(activities)

    per_rep = positive.groupby("replicate").size()
    if not (per_rep == expected).all():
        bad = per_rep[per_rep != expected]
        raise ValueError(
            f"Positive-fraction design is incomplete for {len(bad)} worlds; expected {expected} rows/world."
        )

    per_condition = positive.groupby(["synthetic_fraction", "activity_multiplier"])["replicate"].nunique()
    if per_condition.nunique() != 1:
        raise ValueError("Conditions do not contain the same number of unique worlds after deduplication.")

    return fractions, activities


# ---------- condition summaries ----------

def condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fraction, activity), group in df.groupby(["synthetic_fraction", "activity_multiplier"], sort=True):
        row = {
            "synthetic_fraction": fraction,
            "activity_multiplier": activity,
            "n_worlds": int(group["replicate"].nunique()),
        }
        for metric in SUMMARY_METRICS:
            stats = mean_ci(group[metric])
            for name, value in stats.items():
                row[f"{metric}_{name}"] = value

        if fraction > 0:
            row["P_D_gt_0"] = float(np.mean(group["exposure_minus_production"] > 0))
            row["P_exposure_majority"] = float(np.mean(group["synthetic_exposure_majority"] > 0.5))
            row["P_production_majority"] = float(np.mean(group["synthetic_production_majority"] > 0.5))
        else:
            row["P_D_gt_0"] = math.nan
            row["P_exposure_majority"] = 0.0
            row["P_production_majority"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- paired contrasts ----------

def paired_contrast(
    positive: pd.DataFrame,
    fraction: float,
    treatment: float,
    baseline: float,
    metric: str,
) -> dict:
    subset = positive[positive["synthetic_fraction"] == fraction]
    wide = subset.pivot(index="replicate", columns="activity_multiplier", values=metric)
    if baseline not in wide.columns or treatment not in wide.columns:
        return {}

    diff = (wide[treatment] - wide[baseline]).dropna().to_numpy(dtype=float)
    stats = mean_ci(diff)
    n = len(diff)
    mean = stats["mean"]
    std = stats["std"]
    if n > 1 and std > 0:
        t_stat = mean / (std / math.sqrt(n))
        p = float(2 * t.sf(abs(t_stat), df=n - 1))
        dz = mean / std
    elif n > 1 and std == 0:
        t_stat = math.inf if mean != 0 else 0.0
        p = 0.0 if mean != 0 else 1.0
        dz = math.copysign(math.inf, mean) if mean != 0 else 0.0
    else:
        t_stat = p = dz = math.nan

    return {
        "synthetic_fraction": fraction,
        "metric": metric,
        "baseline_activity": baseline,
        "treatment_activity": treatment,
        "n_worlds": n,
        "mean_difference": mean,
        "std_difference": std,
        "median_difference": stats["median"],
        "ci95_low": stats["ci95_low"],
        "ci95_high": stats["ci95_high"],
        "t_statistic": t_stat,
        "p_raw": p,
        "cohens_dz": dz,
        "fraction_difference_gt_0": float(np.mean(diff > 0)) if n else math.nan,
    }


def build_paired_contrasts(df: pd.DataFrame, fractions: list[float], activities: list[float]) -> pd.DataFrame:
    positive = df[df["synthetic_fraction"] > 0]
    rows = []

    families: dict[str, list[tuple[float, float]]] = {
        "vs_1x": [(1.0, a) for a in activities if a != 1.0],
        "vs_3x": [(3.0, a) for a in activities if a != 3.0] if 3.0 in activities else [],
        "adjacent": list(zip(activities[:-1], activities[1:])),
    }

    for family, comparisons in families.items():
        for fraction in fractions:
            for baseline, treatment in comparisons:
                result = paired_contrast(
                    positive, fraction, treatment, baseline, "exposure_minus_production"
                )
                if result:
                    result["family"] = family
                    rows.append(result)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["p_holm"] = np.nan
    for family, idx in out.groupby("family").groups.items():
        adjusted = holm_adjust(out.loc[idx, "p_raw"].tolist())
        out.loc[idx, "p_holm"] = adjusted
    out["reject_holm_0_05"] = out["p_holm"] < 0.05
    return out.sort_values(["family", "synthetic_fraction", "baseline_activity", "treatment_activity"])


def mechanism_contrasts(df: pd.DataFrame, fractions: list[float], activities: list[float]) -> pd.DataFrame:
    positive = df[df["synthetic_fraction"] > 0]
    metrics = [
        "exposure_efficiency_gap",
        "zero_receipt_gap",
        "p90_receipt_gap",
        "p99_receipt_gap",
        "gini_gap",
        "top_1pct_share_gap",
    ]
    comparisons = []
    if 3.0 in activities:
        comparisons.extend((3.0, a) for a in [5.0, 7.5, 10.0, 15.0] if a in activities)
    comparisons.extend(zip(activities[:-1], activities[1:]))
    comparisons = list(dict.fromkeys(comparisons))

    rows = []
    for metric in metrics:
        for fraction in fractions:
            for baseline, treatment in comparisons:
                result = paired_contrast(positive, fraction, treatment, baseline, metric)
                if result:
                    rows.append(result)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = np.nan
    for metric, idx in out.groupby("metric").groups.items():
        out.loc[idx, "p_holm"] = holm_adjust(out.loc[idx, "p_raw"].tolist())
    out["reject_holm_0_05"] = out["p_holm"] < 0.05
    return out.sort_values(["metric", "synthetic_fraction", "baseline_activity", "treatment_activity"])


# ---------- repeated-measures omnibus Wald tests ----------

def difference_contrast(n_levels: int) -> np.ndarray:
    c = np.zeros((n_levels - 1, n_levels), dtype=float)
    for i in range(1, n_levels):
        c[i - 1, 0] = -1.0
        c[i - 1, i] = 1.0
    return c


def wald_test(mean_vec: np.ndarray, covariance: np.ndarray, n: int, contrast: np.ndarray) -> tuple[float, int, float]:
    theta = contrast @ mean_vec
    variance = contrast @ covariance @ contrast.T / n
    rank = int(np.linalg.matrix_rank(contrast))
    statistic = float(theta.T @ np.linalg.pinv(variance) @ theta)
    p_value = float(chi2.sf(statistic, df=rank))
    return statistic, rank, p_value


def global_wald_tests(df: pd.DataFrame, fractions: list[float], activities: list[float]) -> pd.DataFrame:
    positive = df[df["synthetic_fraction"] > 0]
    columns = pd.MultiIndex.from_product([fractions, activities], names=["fraction", "activity"])
    wide = positive.pivot(index="replicate", columns=["synthetic_fraction", "activity_multiplier"], values="exposure_minus_production")
    wide = wide.reindex(columns=columns)
    if wide.isna().any().any():
        raise ValueError("Cannot run repeated-measures Wald tests: incomplete condition matrix.")

    y = wide.to_numpy(dtype=float)
    n = y.shape[0]
    mean_vec = y.mean(axis=0)
    covariance = np.cov(y, rowvar=False, ddof=1)

    f = len(fractions)
    a = len(activities)
    cf = difference_contrast(f)
    ca = difference_contrast(a)

    contrasts = {
        "synthetic prevalence main effect": np.kron(cf, np.ones((1, a)) / a),
        "activity main effect": np.kron(np.ones((1, f)) / f, ca),
        "prevalence × activity interaction": np.kron(cf, ca),
    }

    rows = []
    for name, contrast in contrasts.items():
        statistic, df_test, p_value = wald_test(mean_vec, covariance, n, contrast)
        rows.append({
            "test": name,
            "wald_chi_square": statistic,
            "df": df_test,
            "p_value": p_value,
            "n_worlds": n,
        })
    return pd.DataFrame(rows)


# ---------- turnover / majority ----------

def interpolate_zero(x1: float, y1: float, x2: float, y2: float) -> float:
    if y2 == y1:
        return (x1 + x2) / 2
    return x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1)


def turnover_summary(summary: pd.DataFrame, fractions: list[float]) -> pd.DataFrame:
    rows = []
    for fraction in fractions:
        s = summary[summary["synthetic_fraction"] == fraction].sort_values("activity_multiplier")
        acts = s["activity_multiplier"].to_numpy(dtype=float)
        means = s["exposure_minus_production_mean"].to_numpy(dtype=float)
        peak_idx = int(np.nanargmax(means))
        peak_activity = acts[peak_idx]
        peak_d = means[peak_idx]

        crossing_low = crossing_high = crossing_estimate = math.nan
        for i in range(peak_idx, len(acts) - 1):
            if means[i] >= 0 and means[i + 1] < 0:
                crossing_low = acts[i]
                crossing_high = acts[i + 1]
                crossing_estimate = interpolate_zero(acts[i], means[i], acts[i + 1], means[i + 1])
                break

        rows.append({
            "synthetic_fraction": fraction,
            "peak_tested_activity": peak_activity,
            "peak_mean_D": peak_d,
            "zero_crossing_lower_tested_activity": crossing_low,
            "zero_crossing_upper_tested_activity": crossing_high,
            "linear_interpolated_zero_crossing": crossing_estimate,
        })
    return pd.DataFrame(rows)


def majority_summary(df: pd.DataFrame, fractions: list[float], activities: list[float]) -> pd.DataFrame:
    positive = df[df["synthetic_fraction"] > 0]
    rows = []
    for (fraction, activity), group in positive.groupby(["synthetic_fraction", "activity_multiplier"], sort=True):
        n = len(group)
        exp_success = int((group["synthetic_exposure_majority"] > 0.5).sum())
        prod_success = int((group["synthetic_production_majority"] > 0.5).sum())
        exp_low, exp_high = wilson_interval(exp_success, n)
        prod_low, prod_high = wilson_interval(prod_success, n)
        rows.append({
            "synthetic_fraction": fraction,
            "activity_multiplier": activity,
            "n_worlds": n,
            "exposure_majority_probability": exp_success / n,
            "exposure_majority_ci95_low": exp_low,
            "exposure_majority_ci95_high": exp_high,
            "production_majority_probability": prod_success / n,
            "production_majority_ci95_low": prod_low,
            "production_majority_ci95_high": prod_high,
        })
    return pd.DataFrame(rows)


# ---------- convergence ----------

def convergence_summary(df: pd.DataFrame, fractions: list[float], activities: list[float], checkpoints: list[int]) -> pd.DataFrame:
    positive = df[df["synthetic_fraction"] > 0]
    n_total = int(positive["replicate"].nunique())
    checkpoints = sorted({n for n in checkpoints if 1 < n <= n_total})
    if n_total not in checkpoints:
        checkpoints.append(n_total)

    final_means = (
        positive.groupby(["synthetic_fraction", "activity_multiplier"])["exposure_minus_production"]
        .mean()
    )

    rows = []
    for n in checkpoints:
        subset = positive[positive["replicate"] < n]
        for (fraction, activity), group in subset.groupby(["synthetic_fraction", "activity_multiplier"], sort=True):
            stats = mean_ci(group["exposure_minus_production"])
            final = float(final_means.loc[(fraction, activity)])
            rows.append({
                "n_worlds": n,
                "synthetic_fraction": fraction,
                "activity_multiplier": activity,
                "mean_D": stats["mean"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
                "ci95_half_width": stats["ci95_half_width"],
                "absolute_difference_from_final_mean": abs(stats["mean"] - final),
            })
    return pd.DataFrame(rows)


# ---------- figures ----------

def plot_metric_with_ci(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    baseline_zero: bool = True,
) -> None:
    rows = summary[summary["synthetic_fraction"] > 0]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for fraction in sorted(rows["synthetic_fraction"].unique()):
        s = rows[rows["synthetic_fraction"] == fraction].sort_values("activity_multiplier")
        x = s["activity_multiplier"].to_numpy(dtype=float)
        y = s[f"{metric}_mean"].to_numpy(dtype=float)
        low = s[f"{metric}_ci95_low"].to_numpy(dtype=float)
        high = s[f"{metric}_ci95_high"].to_numpy(dtype=float)
        line, = ax.plot(x, y, marker="o", label=f"{fraction:.0%} synthetic")
        ax.fill_between(x, low, high, alpha=0.10, color=line.get_color())

    if baseline_zero:
        ax.axhline(0.0, linestyle="--", linewidth=1)
    activities = sorted(rows["activity_multiplier"].unique())
    ax.set_xscale("log")
    ax.set_xticks(activities, labels=[activity_label(x) for x in activities])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Synthetic activity multiplier")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_paired_vs_one(paired: pd.DataFrame, path: Path) -> None:
    data = paired[paired["family"] == "vs_1x"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for fraction in sorted(data["synthetic_fraction"].unique()):
        s = data[data["synthetic_fraction"] == fraction].sort_values("treatment_activity")
        x = s["treatment_activity"].to_numpy(dtype=float)
        y = s["mean_difference"].to_numpy(dtype=float)
        low = s["ci95_low"].to_numpy(dtype=float)
        high = s["ci95_high"].to_numpy(dtype=float)
        line, = ax.plot(x, y, marker="o", label=f"{fraction:.0%} synthetic")
        ax.fill_between(x, low, high, alpha=0.10, color=line.get_color())
    ax.axhline(0.0, linestyle="--", linewidth=1)
    activities = sorted(data["treatment_activity"].unique())
    ax.set_xscale("log")
    ax.set_xticks(activities, labels=[activity_label(x) for x in activities])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Synthetic activity multiplier")
    ax.set_ylabel(r"Change in $D$ relative to 1×")
    ax.set_title("Paired effect of increased synthetic activity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_heatmap(summary: pd.DataFrame, fractions: list[float], activities: list[float], path: Path) -> None:
    matrix = np.full((len(fractions), len(activities)), np.nan)
    for i, fraction in enumerate(fractions):
        for j, activity in enumerate(activities):
            row = summary[
                (summary["synthetic_fraction"] == fraction)
                & (summary["activity_multiplier"] == activity)
            ]
            if len(row):
                matrix[i, j] = row.iloc[0]["exposure_minus_production_mean"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(activities)), labels=[activity_label(x) for x in activities])
    ax.set_yticks(range(len(fractions)), labels=[f"{x:.0%}" for x in fractions])
    ax.set_xlabel("Synthetic activity multiplier")
    ax.set_ylabel("Synthetic producer prevalence")
    ax.set_title(r"Mean $D$: exposure share − production share")
    for i in range(len(fractions)):
        for j in range(len(activities)):
            ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Mean D")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_convergence(convergence: pd.DataFrame, path: Path) -> None:
    by_n = convergence.groupby("n_worlds").agg(
        max_abs_mean_difference=("absolute_difference_from_final_mean", "max"),
        max_ci95_half_width=("ci95_half_width", "max"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(by_n["n_worlds"], by_n["max_abs_mean_difference"], marker="o", label="Max |mean − final mean|")
    ax.plot(by_n["n_worlds"], by_n["max_ci95_half_width"], marker="o", label="Max 95% CI half-width")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of independent worlds")
    ax.set_ylabel("Maximum deviation")
    ax.set_title("Monte Carlo convergence of D estimates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


# ---------- text report ----------

def write_report(
    path: Path,
    input_path: Path,
    input_hash: str,
    validation: dict,
    global_tests: pd.DataFrame,
    turnover: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    lines = [
        "Deterministic volume-sweep analysis",
        "===================================",
        f"Input: {input_path}",
        f"SHA-256: {input_hash}",
        "",
        "Validation / cleaning",
        "---------------------",
    ]
    for key, value in validation.items():
        lines.append(f"{key}: {value}")

    lines.extend(["", "Repeated-measures omnibus Wald tests", "------------------------------------"])
    for _, row in global_tests.iterrows():
        lines.append(
            f"{row['test']}: chi2({int(row['df'])})={row['wald_chi_square']:.3f}, p={row['p_value']:.6g}"
        )

    lines.extend(["", "Turnover summary", "----------------"])
    for _, row in turnover.iterrows():
        lines.append(
            f"{row['synthetic_fraction']:.0%}: peak tested activity={row['peak_tested_activity']:g}x, "
            f"peak mean D={row['peak_mean_D']:+.6f}, zero crossing between "
            f"{row['zero_crossing_lower_tested_activity']:g}x and "
            f"{row['zero_crossing_upper_tested_activity']:g}x "
            f"(linear interpolation={row['linear_interpolated_zero_crossing']:.3f}x)"
        )

    lines.extend(["", "Key paired contrasts (Holm-adjusted)", "-----------------------------------"])
    key = paired[
        (paired["family"] == "vs_3x")
        & (paired["treatment_activity"].isin([7.5, 10.0, 15.0]))
    ]
    for _, row in key.iterrows():
        lines.append(
            f"{row['synthetic_fraction']:.0%}, {row['treatment_activity']:g}x − 3x: "
            f"mean ΔD={row['mean_difference']:+.6f}, 95% CI "
            f"[{row['ci95_low']:+.6f}, {row['ci95_high']:+.6f}], "
            f"dz={row['cohens_dz']:+.3f}, Holm p={row['p_holm']:.6g}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic final analysis for the DMAS synthetic-volume sweep."
    )
    parser.add_argument("--input", type=Path, required=True, help="runs.csv or runs.csv.gz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/volume_sweep_big/final_analysis"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures/volume_sweep_big/final_analysis"),
    )
    parser.add_argument(
        "--write-cleaned",
        action="store_true",
        help="Also write the deduplicated run-level dataset as cleaned_runs.csv.gz.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Hashing input: {args.input}")
    input_hash = sha256_file(args.input)
    print(f"SHA-256: {input_hash}")

    print("Loading and validating run-level data...")
    df, validation = load_runs(args.input)
    fractions, activities = validate_balanced_design(df)

    print("=== Cleaning / design validation ===")
    for key, value in validation.items():
        print(f"{key}: {value}")
    print(f"fractions: {fractions}")
    print(f"activities: {activities}")

    if args.write_cleaned:
        cleaned_path = args.output_dir / "cleaned_runs.csv.gz"
        print(f"Writing deduplicated runs: {cleaned_path}")
        df.to_csv(cleaned_path, index=False, compression="gzip")

    print("Computing condition summaries...")
    summary = condition_summary(df)
    save_csv(summary, args.output_dir / "condition_summary.csv")

    print("Computing paired contrasts and Holm corrections...")
    paired = build_paired_contrasts(df, fractions, activities)
    save_csv(paired, args.output_dir / "paired_contrasts.csv")

    print("Computing mechanism contrasts...")
    mechanisms = mechanism_contrasts(df, fractions, activities)
    save_csv(mechanisms, args.output_dir / "mechanism_contrasts.csv")

    print("Computing repeated-measures omnibus tests...")
    global_tests = global_wald_tests(df, fractions, activities)
    save_csv(global_tests, args.output_dir / "global_wald_tests.csv")

    print("Characterising turnover...")
    turnover = turnover_summary(summary, fractions)
    save_csv(turnover, args.output_dir / "turnover_summary.csv")

    print("Computing majority-exposure probabilities...")
    majority = majority_summary(df, fractions, activities)
    save_csv(majority, args.output_dir / "majority_summary.csv")

    print("Computing convergence diagnostics...")
    convergence = convergence_summary(
        df, fractions, activities, checkpoints=[100, 500, 1000, 2000, 5000, 10000]
    )
    save_csv(convergence, args.output_dir / "convergence.csv")

    duplicate_report = pd.DataFrame([validation])
    save_csv(duplicate_report, args.output_dir / "data_validation.csv")

    print("Generating figures...")
    plot_metric_with_ci(
        summary,
        "exposure_minus_production",
        "Exposure share − production share",
        "Synthetic-content exposure relative to production",
        args.figures_dir / "D_vs_activity.png",
    )
    plot_paired_vs_one(paired, args.figures_dir / "paired_effect_vs_1x.png")
    plot_metric_with_ci(
        summary,
        "exposure_efficiency_gap",
        "Synthetic − human receipts per post",
        "Relative per-post exposure efficiency",
        args.figures_dir / "exposure_efficiency_gap.png",
    )
    plot_metric_with_ci(
        summary,
        "zero_receipt_gap",
        "Synthetic − human zero-receipt fraction",
        "Difference in zero-receipt post probability",
        args.figures_dir / "zero_receipt_gap.png",
    )
    plot_heatmap(summary, fractions, activities, args.figures_dir / "D_heatmap.png")
    plot_convergence(convergence, args.figures_dir / "convergence.png")

    write_report(
        args.output_dir / "analysis_report.txt",
        args.input,
        input_hash,
        validation,
        global_tests,
        turnover,
        paired,
    )

    print("\nDone.")
    print(f"Tables:  {args.output_dir}")
    print(f"Figures: {args.figures_dir}")
    print(f"Report:  {args.output_dir / 'analysis_report.txt'}")


if __name__ == "__main__":
    main()
