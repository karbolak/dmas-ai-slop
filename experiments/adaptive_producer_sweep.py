"""Paired content-adaptation experiment; run from repository root as a module.

Refuses existing output directories. Saves complete configuration, immutable run
fingerprints, per-world run rows and per-policy strategy diagnostics. Feedback is
cumulative provenance-attributed receipts per accepted original post, including
receipts delivered later by human reposters. No future receipt is used at selection.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import time

import doces
import doces_core
import numpy as np
import pandas as pd
from experiments.alpha_volume_sweep import (
    analyse_run, expected_activation_share, generate_directed_er_network,
    generate_initial_opinions, generate_producer_order, generate_replicate_seeds,
    get_synthetic_users, load_config, safe_ratio,
)


def condition_grid(cfg):
    grid=[]
    for block in cfg['design']:
        grid.extend(itertools.product(block['feed_sizes'], block['fractions'],
                                      block['activity_multipliers'], cfg['producer']['policies']))
    if len(grid)!=len(set(grid)):
        raise ValueError('Duplicate experimental conditions')
    return grid


def run_condition(cfg, info, edges, opinions, order, condition):
    feed,fraction,multiplier,policy=condition
    started=time.perf_counter()
    n=cfg['network']['n_agents']
    users=get_synthetic_users(order,fraction)
    selected=np.array(sorted(users),dtype=int)
    model=doces.Opinion_dynamics(n,edges,cfg['network']['directed'],verbose=False)
    model.set_activity_weights([multiplier if i in users else 1. for i in range(n)])
    mode={'baseline':0,'random':1,'adaptive':2}[policy]
    model.set_producer_policies([mode if i in users else 0 for i in range(n)],
                               cfg['producer']['n_strategies'],cfg['producer']['epsilon'])
    sim=cfg['simulation']
    model.simulate_dynamics(number_of_iterations=sim['iterations'],phi=sim['phi'],mu=sim['mu'],
        posting_filter=sim['posting_filter'],receiving_filter=sim['receiving_filter'],
        b=list(opinions),feed_size=feed,rewire=sim['rewire'],delta=sim['delta'],
        min_opinion=-1.,max_opinion=1.,verbose=False,rand_seed=info['simulation_seed'])
    row=dict(info,feed_size=feed,synthetic_fraction=fraction,activity_multiplier=multiplier,
             policy=policy,expected_activation_share=expected_activation_share(n,len(users),multiplier))
    row.update(analyse_run(model,users,True))
    stats={name:np.asarray(values,dtype=np.int64) for name,values in model.producer_stats.items()}
    # Reconcile all accepted originals and all provenance-attributed receipts.
    assert int(stats['accepted'].sum())==row['synthetic_posts']+row['human_posts']
    assert int(stats['receipts'].sum())==row['synthetic_exposures']+row['human_exposures']
    attempts=int(stats['attempts'][selected].sum())
    row['synthetic_creation_attempts']=attempts
    row['human_creation_attempts']=int(stats['attempts'].sum())-attempts
    row['synthetic_acceptance_rate']=safe_ratio(row['synthetic_posts'],attempts)
    row['human_acceptance_rate']=safe_ratio(row['human_posts'],row['human_creation_attempts'])
    row['synthetic_receipts_per_attempt']=safe_ratio(row['synthetic_exposures'],attempts)
    row['human_receipts_per_attempt']=safe_ratio(row['human_exposures'],row['human_creation_attempts'])
    early=stats['early_attempts'][selected]
    late=stats['late_attempts'][selected]
    observed=(early.sum(axis=1)>0)&(late.sum(axis=1)>0)
    row['producers_observed_both_quarters']=int(observed.sum())
    for label,values in [('early',early),('late',late)]:
        total=values[observed].sum(axis=1)
        row[f'{label}_mean_max_strategy_fraction']=float(np.mean(values[observed].max(axis=1)/total)) if len(total) else np.nan
    row['concentration_change']=row['late_mean_max_strategy_fraction']-row['early_mean_max_strategy_fraction']
    for name,values in stats.items():
        for arm,total in enumerate(values[selected].sum(axis=0)):
            row[f'synthetic_{name}_arm_{arm}']=int(total)
    row['elapsed_seconds']=time.perf_counter()-started
    return row


def run_experiment(cfg,output):
    grid=condition_grid(cfg)
    seeds=generate_replicate_seeds(cfg['seeds']['master'],cfg['seeds']['replicates'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    pd.DataFrame(seeds).to_csv(output/'seed_manifest.csv',index=False)
    root=Path(__file__).resolve().parents[1]
    engine=Path(doces.__file__).resolve().parents[1]
    def revision(path):
        return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    files=[Path(__file__),root/'experiments/alpha_volume_sweep.py',
           engine/'doces/doces.py',*sorted((engine/'doces/Source').glob('*.[ch]')),
           *sorted((engine/'doces/Python').glob('*.[ch]')),Path(doces_core.__file__)]
    metadata=dict(started_utc=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(),numpy=np.__version__,
                  research_revision=revision(root),simulator_revision=revision(engine),
                  n_conditions=len(grid),replicates=len(seeds),
                  hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    rows=[]
    for info in seeds:
        n=cfg['network']['n_agents']
        edges=generate_directed_er_network(n,cfg['network']['mean_in_degree'],info['network_seed'])
        opinions=generate_initial_opinions(n,info['opinion_seed'])
        order=generate_producer_order(n,info['producer_assignment_seed'])
        world=[run_condition(cfg,info,edges,opinions,order,condition) for condition in grid]
        pd.DataFrame(world).to_csv(output/'runs.csv',mode='a',header=not rows,index=False)
        rows.extend(world)
        if (info['replicate']+1)%10==0:
            print(f"Completed {info['replicate']+1}/{len(seeds)} worlds ({len(rows)} runs)",flush=True)
    metadata.update(completed_utc=datetime.now(timezone.utc).isoformat(),runs=len(rows),
                    runs_sha256=hashlib.sha256((output/'runs.csv').read_bytes()).hexdigest())
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return pd.DataFrame(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/adaptive_producers.yaml'))
    parser.add_argument('--output',type=Path,default=Path('results/adaptive_producers'))
    args=parser.parse_args()
    run_experiment(load_config(args.config),args.output)
