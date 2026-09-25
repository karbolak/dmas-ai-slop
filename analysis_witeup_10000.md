# Volume Sweep Results Summary

## Experiment

The final volume sweep tested how synthetic-producer prevalence and relative activity affect the
relationship between synthetic-content production and exposure.

The experiment used:

- 10,000 independently generated simulation worlds;
- synthetic-producer prevalence levels of 1%, 5%, 10%, 15%, 20%, and 25%;
- synthetic activity multipliers of 1×, 2×, 3×, 5×, 7.5×, 10×, and 15×;
- one 0% synthetic control condition;
- 43 experimental conditions in total;
- 430,000 unique condition-runs after deterministic deduplication.

The main outcome was

\[
D = S_{\mathrm{exposure}} - S_{\mathrm{production}},
\]

where:

- \(S_{\mathrm{production}}\) is the share of experimental posts originating from synthetic producers;
- \(S_{\mathrm{exposure}}\) is the share of successful feed receipts attributable to synthetic-origin posts.

Interpretation:

- \(D > 0\): synthetic content receives disproportionate exposure relative to production;
- \(D = 0\): exposure is proportional to production;
- \(D < 0\): synthetic content is attenuated relative to production.

The final deterministic analysis is implemented in:

`analysis/analyze_volume_sweep_final.py`

Primary outputs are stored under:

`results/volume_sweep_big/final_analysis/`

and figures under:

`figures/volume_sweep_big/final_analysis/`

---

## Data validation

The raw run file contained duplicated condition-runs produced by the checkpointed execution process.
The deterministic analysis identified:

| Quantity | Value |
|---|---:|
| Raw rows | 858,916 |
| Unique condition-runs | 430,000 |
| Duplicate rows removed | 428,916 |
| Independent worlds | 10,000 |
| Experimental conditions | 43 |
| Duplicate keys with inconsistent core metrics | 0 |

The duplicates were therefore exact repetitions for the core simulation outcomes rather than conflicting
simulation results. The final analysis deduplicates by:

`(replicate, synthetic_fraction, activity_multiplier)`

before calculating confidence intervals or statistical tests.

See:

- `results/volume_sweep_big/final_analysis/data_validation.csv`
- `results/volume_sweep_big/final_analysis/analysis_report.txt`

Weighted activation also closely matched the intended activation probabilities across conditions.

See:

- `figures/volume_sweep_big/10000/activation_validation.png`

---

## Main result: the effect of activity is non-monotonic

Increasing synthetic-producer activity did **not** produce a monotonic increase in synthetic exposure
relative to production.

Across all tested prevalence levels, the same broad pattern appeared:

1. At 1× activity, \(D\) was approximately zero.
2. At 2×–3× activity, \(D\) became slightly positive in most prevalence conditions.
3. Around 5× activity, \(D\) returned to approximately zero.
4. At 7.5×, 10×, and 15× activity, \(D\) became increasingly negative.

The strongest positive effects were small, while the attenuation at high activity was considerably larger.

### Estimated turnover

| Synthetic prevalence | Peak tested activity | Peak mean \(D\) | Estimated zero crossing |
|---:|---:|---:|---:|
| 1% | 2× | +0.0004 | 4.84× |
| 5% | 3× | +0.0018 | 5.28× |
| 10% | 3× | +0.0019 | 5.11× |
| 15% | 3× | +0.0034 | 5.00× |
| 20% | 3× | +0.0037 | 4.87× |
| 25% | 2× | +0.0036 | 4.67× |

The estimated transition from over-exposure to under-exposure is therefore tightly clustered around
approximately **5× activity** across the full 1%–25% prevalence range.

See:

- `figures/volume_sweep_big/final_analysis/D_vs_activity.png`
- `figures/volume_sweep_big/final_analysis/D_heatmap.png`
- `results/volume_sweep_big/final_analysis/turnover_summary.csv`

---

## Statistical evidence

Repeated-measures omnibus tests found clear effects of both experimental factors and their interaction:

| Effect | Wald test |
|---|---|
| Synthetic prevalence | \(\chi^2(5)=869.53\), \(p \approx 1.0\times10^{-185}\) |
| Activity multiplier | \(\chi^2(6)=29176.50\), \(p <\) machine precision |
| Prevalence × activity | \(\chi^2(30)=4042.40\), \(p <\) machine precision |

Because the experiment contains 10,000 worlds, statistical significance alone is not the main
interpretive criterion. Effect sizes and confidence intervals are more informative.

Paired within-world contrasts show that the decline from moderate activity to high activity is highly
consistent. For example:

| Prevalence | Contrast | Mean \(\Delta D\) | 95% CI | \(d_z\) |
|---:|---|---:|---|---:|
| 5% | 15× − 3× | -0.0327 | [-0.0336, -0.0318] | -0.721 |
| 10% | 15× − 3× | -0.0320 | [-0.0329, -0.0311] | -0.685 |
| 15% | 15× − 3× | -0.0287 | [-0.0296, -0.0278] | -0.614 |
| 20% | 15× − 3× | -0.0244 | [-0.0252, -0.0235] | -0.539 |
| 25% | 15× − 3× | -0.0196 | [-0.0205, -0.0187] | -0.444 |

All reported high-activity contrasts remain supported after Holm correction.

See:

- `results/volume_sweep_big/final_analysis/global_wald_tests.csv`
- `results/volume_sweep_big/final_analysis/paired_contrasts.csv`
- `figures/volume_sweep_big/final_analysis/paired_effect_vs_1x.png`

---

## Per-post exposure efficiency

The high-activity downturn in \(D\) is accompanied by a strong reduction in the exposure obtained per
synthetic post.

Define per-post exposure efficiency as:

\[
E_g =
\frac{\text{successful receipts of posts from group }g}
     {\text{number of posts produced by group }g}.
\]

The diagnostic quantity plotted is:

\[
E_{\mathrm{synthetic}} - E_{\mathrm{human}}.
\]

At moderate activity, synthetic posts receive approximately the same or slightly more exposure per post
than human posts.

At high activity, this relationship reverses. The difference becomes increasingly negative as activity
increases from 7.5× to 15×.

This provides a direct explanation of the sign of \(D\): in the cleaned dataset, the sign of \(D\)
matches the sign of the synthetic-minus-human per-post exposure-efficiency difference in all analysed
conditions.

See:

- `figures/volume_sweep_big/final_analysis/exposure_efficiency_gap.png`
- `results/volume_sweep_big/final_analysis/mechanism_contrasts.csv`

This is best described as a **mechanistic indicator**, not yet proof of the underlying causal mechanism.

---

## Zero-receipt posts

The probability that a post receives no downstream feed receipts also changes with synthetic activity.

At approximately 2×–3× activity, synthetic posts are generally slightly less likely than human posts
to receive zero receipts.

At high activity, particularly for larger synthetic populations, this pattern reverses. Synthetic posts
become increasingly more likely than human posts to receive no downstream exposure.

See:

`figures/volume_sweep_big/final_analysis/zero_receipt_gap.png`

This pattern is consistent with increasing competition among synthetic posts for limited diffusion
opportunities or feed space. However, the current measurements do not directly distinguish between
feed-capacity effects, receiving-filter effects, repost dynamics, or other mechanisms. Therefore,
"self-competition" or "congestion" should currently be treated as hypotheses rather than established
causal explanations.

---

## Production dominance is not the same as exposure amplification

A second important result is the distinction between:

1. **synthetic-content dominance**, where synthetic content constitutes a large or majority share of
   exposure; and
2. **disproportionate amplification**, where synthetic exposure exceeds its share of production.

These are not equivalent.

At sufficiently high activity, a small synthetic population can produce a very large share of all posts
and consequently account for a large share of exposure. However, \(D\) can simultaneously be negative.

In other words, synthetic content can dominate the feed through **production volume alone**, even while
each unit of synthetic production receives less exposure than expected from proportional allocation.

This distinction is important for interpreting synthetic-content saturation: a feed can become dominated
by synthetic-origin content without requiring an algorithmic amplification advantage.

See:

- `results/volume_sweep_big/final_analysis/majority_summary.csv`
- `results/volume_sweep_big/final_analysis/condition_summary.csv`

---

## Interpretation

The final sweep does not support a simple model in which increasing synthetic-producer activity
continuously increases disproportionate synthetic exposure.

Instead, the simulations show a robust turnover:

> Moderate synthetic activity can produce a small exposure advantage, but beyond approximately 5×
> relative activity, additional production increasingly reduces exposure per synthetic post and causes
> synthetic content to become under-exposed relative to its production share.

The effect is present across synthetic-producer prevalence levels from 1% to 25%, although its magnitude
depends on prevalence.

This suggests diminishing and eventually negative returns to extreme synthetic production within the
current adaptive social-network model.

The most plausible current explanation is that very high production creates competition among
synthetic-origin posts for limited diffusion opportunities. The per-post exposure and zero-receipt
diagnostics are consistent with this interpretation, but additional simulator instrumentation would be
required to determine whether the effect is specifically caused by feed replacement, receiving-filter
rejection, repost competition, rewiring, or another process.

---

## Implications for the research question

The results motivate a research question centred on the transition between amplification and attenuation,
rather than a simple question of whether greater synthetic activity increases exposure.

A current candidate is:

> **How does synthetic-producer activity affect exposure relative to production, and under what
> conditions does increasing activity shift from amplification to attenuation?**

A complementary subquestion is:

> **When does synthetic-content dominance arise from production volume rather than disproportionate
> exposure?**

These questions distinguish between:

- how much synthetic content is produced;
- how much synthetic content users receive;
- whether exposure is proportional to production;
- and whether extreme activity creates diminishing exposure returns.

---

## Main reproducibility files

### Simulation data

- `results/volume_sweep_big/runs.csv`
- `results/volume_sweep_big/seed_manifest.csv`
- `configs/volume_sweep_big.yaml`

### Deterministic analysis

- `analysis/analyze_volume_sweep_final.py`
- `results/volume_sweep_big/final_analysis/analysis_report.txt`
- `results/volume_sweep_big/final_analysis/data_validation.csv`
- `results/volume_sweep_big/final_analysis/condition_summary.csv`
- `results/volume_sweep_big/final_analysis/global_wald_tests.csv`
- `results/volume_sweep_big/final_analysis/paired_contrasts.csv`
- `results/volume_sweep_big/final_analysis/mechanism_contrasts.csv`
- `results/volume_sweep_big/final_analysis/turnover_summary.csv`
- `results/volume_sweep_big/final_analysis/majority_summary.csv`
- `results/volume_sweep_big/final_analysis/convergence.csv`

### Main figures

- `figures/volume_sweep_big/final_analysis/D_vs_activity.png`
- `figures/volume_sweep_big/final_analysis/D_heatmap.png`
- `figures/volume_sweep_big/final_analysis/paired_effect_vs_1x.png`
- `figures/volume_sweep_big/final_analysis/exposure_efficiency_gap.png`
- `figures/volume_sweep_big/final_analysis/zero_receipt_gap.png`
- `figures/volume_sweep_big/final_analysis/convergence.png`

---

## Current takeaway

The central result of the volume experiment is not that high-volume synthetic producers automatically
receive disproportionate exposure.

Instead:

**Synthetic activity exhibits a non-monotonic relationship with relative exposure. Moderate activity
produces small positive exposure effects, while extreme activity produces increasingly strong attenuation
relative to production. At the same time, sufficiently high production volume can still make synthetic
content dominate overall exposure.**