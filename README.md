# DMAS: producer activity and relative exposure

Research code and final report for the University of Groningen Design of Multi-Agent Systems project. The simulator is in the sibling `doces-slop` repository.

The implemented research question is: **How do producer activity and prevalence affect a labelled subgroup's share of feed receipts relative to its share of production in DOCES?**

High-volume AI-generated content motivates the question. The model represents a fixed activity advantage, producer prevalence, and post provenance. It does not represent content quality, generative models, coordinated ownership, or producers adapting to engagement rewards. Adaptive follower rewiring is distinct from adaptive production strategy.

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

Adaptive producer strategies, alternative networks, receiving rules, innovation rates, longer horizons, and initialization controls require separate experiments. A complete model of AI-slop production would also need an explicit reward and strategy-update rule.
