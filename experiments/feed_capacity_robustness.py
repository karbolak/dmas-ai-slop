"""Focused paired feed-capacity sensitivity sweep; reuses the frozen volume model.

Run from the research repository: python -m experiments.feed_capacity_robustness
Outputs use a fresh directory; existing evidence is never appended or overwritten.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import doces
import doces_core
import numpy as np
import pandas as pd
from scipy.stats import t

from experiments.alpha_volume_sweep import (
    generate_directed_er_network, generate_initial_opinions,
    generate_producer_order, generate_replicate_seeds, load_config, run_condition,
)


def summarize(frame):
    def estimate(values):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all() or len(values) < 2:
            raise ValueError('Each estimate requires at least two finite paired worlds')
        half = t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values))
        return dict(n=len(values), mean=values.mean(), ci95_low=values.mean()-half,
                    ci95_high=values.mean()+half)

    keys = ['feed_size', 'synthetic_fraction', 'activity_multiplier']
    summary = [dict(zip(keys, key), **estimate(group.exposure_minus_production))
               for key, group in frame.groupby(keys)]
    contrasts = []
    for (feed, fraction), group in frame.groupby(keys[:2]):
        paired = group.pivot(index='replicate', columns='activity_multiplier',
                             values='exposure_minus_production')
        contrasts.append(dict(feed_size=feed, synthetic_fraction=fraction,
                              contrast='15x minus 3x', **estimate(paired[15.]-paired[3.])))
    return pd.DataFrame(summary), pd.DataFrame(contrasts)


def run_experiment(cfg, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    root = Path(__file__).resolve().parents[1]
    def revision(path):
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    sources = [Path(__file__), root/'experiments/alpha_volume_sweep.py',
               *sorted((Path(doces.__file__).parent/'Source').glob('*.[ch]')),
               *sorted((Path(doces.__file__).parent/'Python').glob('*.[ch]')),
               Path(doces.__file__).parent/'doces.py', Path(doces_core.__file__)]
    metadata = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), numpy=np.__version__,
                    research_revision=revision(root),
                    simulator_revision=revision(Path(doces.__file__).parent),
                    simulator_path=doces.__file__, binary_path=doces_core.__file__,
                    sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    note='Uncommitted runner/config recorded by hashes and config.json; pointwise t intervals, no multiplicity adjustment.')
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    seeds = generate_replicate_seeds(cfg['seeds']['master'], cfg['seeds']['replicates'])
    pd.DataFrame(seeds).to_csv(output/'seed_manifest.csv', index=False)
    rows = []
    n = cfg['network']['n_agents']
    for info in seeds:
        world_start = len(rows)
        edges = generate_directed_er_network(n, cfg['network']['mean_in_degree'], info['network_seed'])
        opinions = generate_initial_opinions(n, info['opinion_seed'])
        order = generate_producer_order(n, info['producer_assignment_seed'])
        for feed in cfg['robustness']['feed_sizes']:
            condition = copy.deepcopy(cfg)
            condition['simulation']['feed_size'] = feed
            for fraction in cfg['synthetic']['fractions']:
                for multiplier in cfg['synthetic']['activity_multipliers']:
                    row = run_condition(condition, edges, opinions, order, info, fraction, multiplier)
                    row['feed_size'] = feed
                    rows.append(row)
        if (info['replicate']+1) % 20 == 0:
            print(f"Completed {info['replicate']+1}/{len(seeds)} worlds", flush=True)
        pd.DataFrame(rows[world_start:]).to_csv(output/'runs.csv', mode='a',
                                      header=info['replicate'] == 0, index=False)
    frame = pd.DataFrame(rows)
    summary, paired = summarize(frame)
    summary.to_csv(output/'summary.csv', index=False)
    paired.to_csv(output/'paired_activity_effects.csv', index=False)
    metadata['completed_utc'] = datetime.now(timezone.utc).isoformat()
    metadata['condition_runs'] = len(frame)
    metadata['runs_sha256'] = hashlib.sha256((output/'runs.csv').read_bytes()).hexdigest()
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return frame


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/feed_capacity_robustness.yaml'))
    parser.add_argument('--output', type=Path, default=Path('results/feed_capacity_robustness'))
    args = parser.parse_args()
    run_experiment(load_config(args.config), args.output)


if __name__ == '__main__':
    main()
