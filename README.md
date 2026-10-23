# DMAS: producer activity, content adaptation and relative exposure

Research code and final report for the University of Groningen Design of Multi-Agent Systems project. The simulator is in the sibling `doces-slop` repository. The final paper can be found [here](./dmas-ai-slop-paper.pdf)

The extended research question is: **How do synthetic-producer activity and receipt-based content adaptation affect relative exposure in an adaptive social-network simulation?**

High-volume AI-generated content motivates the question. The model represents a fixed activity advantage, producer prevalence, post provenance, and a specified receipt-responsive opinion-selection policy. It does not represent actual media quality, generative models, coordinated ownership, or engagement rewards. Producer adaptation is separate from adaptive follower rewiring.

## Layout

- `experiments/alpha_volume_sweep.py`: original independent-world volume sweep.
- `configs/volume_sweep_big.yaml`: original 10,000-world, 43-condition design.
- `analysis/analyze_volume_sweep_final.py`: deduplication, statistical analysis and figures.
- `experiments/feed_capacity_robustness.py`: focused paired capacity sensitivity experiment.
- `configs/feed_capacity_robustness.yaml`: 200 worlds, capacities 1/5/20, prevalences 5%/25%, activities 1×/3×/15×.
- `analysis/plot_feed_capacity.py`: regenerate the sensitivity figure and report table.
- `results/feed_capacity_robustness/`: full new run data, configuration, seed manifest, summaries and provenance hashes.
- `results/beta_revalidation/`: fresh analysis of the original run-level data.
- `report/DMAS-5-final.tex`: editable report; see `report/README.md` for compilation.

## Environment and validation

Use Python 3.12 and the existing Conda specification, which pins NumPy below 2. Install the sibling simulator in that environment, or compile it in place and put its directory on `PYTHONPATH`:

```sh
conda env create -f environment.yml
conda activate dmas-ai-slop
pip install -e ../doces-slop
python scripts/check_setup.py
python scripts/check_weighted_activation.py
python scripts/check_content_provenance.py
python -m unittest discover -s tests -v
```

The simulator's own `baseline_smoke_test.py` and `tests/test_feed_receipts.py` provide additional checks. Existing native build warnings are documented in its `VALIDATION.md`.

## Reproduce the feed-capacity sensitivity experiment

Run from this repository:

```sh
python -m experiments.feed_capacity_robustness
python analysis/plot_feed_capacity.py
```

The output directory must not already exist. Use `--output results/another_name` for an independent rerun. The runner deliberately refuses to append to or overwrite prior evidence. Interrupted outputs remain available for inspection; there is no automatic resume.

Each world shares its graph, initial opinions, nested producer identities and simulation seed across all 18 conditions. The new master seed is 20261022. All other simulation parameters match the original main sweep. A request for 100,000 iterations executes 100,001 activations under the inherited loop convention, which is retained for comparability.

The new sensitivity analysis uses pointwise Student's t intervals and paired 15× minus 3× activity contrasts. It is exploratory and does not estimate a new zero crossing. Changing feed capacity also changes initialization, repost order and duplicate suppression, so it is not an isolated test of attention scarcity.

## Reanalyse the main sweep

The original raw data are distributed in `volume_sweep_big_results.tar.gz` and retained in the original local project checkout. Inspect archive paths before extracting to a fresh directory. Provide the extracted `runs.csv` to:

```sh
python analysis/analyze_volume_sweep_final.py \
  --input /path/to/runs.csv \
  --output-dir results/beta_revalidation \
  --figures-dir figures/beta_revalidation
```

The fresh analysis removes 428,916 duplicate rows from 858,916 raw rows and retains 430,000 unique condition-runs from 10,000 independent worlds. Checked core outcomes agree across duplicate keys. The main sweep's precision describes Monte Carlo uncertainty within the specified model, not empirical accuracy on a social-media platform.

## Interpretation and further work

Production counts accepted distinct innovations; exposure counts accepted follower feed insertions of posts absent from the receiving feed. Background initialization posts and their receipts are excluded from group metrics but remain active in the dynamics. Receipts are not unique audience reach, impressions or engagement.

More complex producer strategies, alternative networks, receiving rules, innovation rates, longer horizons, and initialization controls require separate experiments. A complete model of AI-slop production would also need an explicit reward and strategy-update rule.

## Adaptive-producer extension

The validated report above remains the backup on `backup/dmas-5-validated` in both repositories. The `feature/adaptive-producers` branch adds a separate follow-up:

```sh
python -m experiments.adaptive_producer_sweep
python -m analysis.analyze_adaptive_producers
```

Use the simulator from the matching feature branch and a fresh output directory. The configuration is `configs/adaptive_producers.yaml`: 200 worlds and 36 conditions (7,200 runs). It compares baseline, random-bin and epsilon-greedy content policies at matched activity. The main grid uses a one-item feed, 5%/10% prevalence and activity 1×/2×/5×/10×. Additional feed capacities 3/5 use 5% prevalence and activity 2×/10×.

Synthetic adaptive producers choose among five opinion bins with epsilon=.1, using cumulative provenance-attributed receipts per accepted original post. The experiment separately records creation attempts, posting acceptance, receipts per accepted post, receipts per attempt, zero-receipt frequency and early/late strategy concentration. Scores include only feedback observed so far; later receipts via human reposts credit the original producer. The policy does not optimize actual media quality, semantic formats, money, or engagement.

`report/DMAS-6-adaptive.tex` is the extended report. Its own bibliography is `report/bibliography-adaptive.bib`; the backup source and bibliography remain separate. The analysis saves full pointwise intervals and adjusts the 24 D policy comparisons together using Holm's method. Eight high-minus-moderate activity interactions form a separate Holm family. Diagnostic effects remain descriptive.

Analysis checks the raw-data fingerprint on every run. It verifies source/binary fingerprints when their recorded files are available and explicitly reports unavailable run-machine files on another computer. Missing local binaries do not prevent reanalysis of the saved data. `python scripts/verify_adaptive_results.py` independently checks the exported estimates and multiplicity corrections without requiring the simulator binary.
