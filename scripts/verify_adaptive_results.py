"""Independent paired-estimate and multiplicity audit for the completed extension."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t
root=Path(__file__).resolve().parents[1]
directory=root/'results/adaptive_producers'
runs=pd.read_csv(directory/'runs.csv')
effects=pd.read_csv(directory/'analysis/paired_policy_effects.csv')
interactions=pd.read_csv(directory/'analysis/activity_interactions.csv')
assert len(runs)==7200
assert not runs.duplicated(['replicate','feed_size','synthetic_fraction','activity_multiplier','policy']).any()
assert (runs.groupby(['feed_size','synthetic_fraction','activity_multiplier','policy']).size()==200).all()
for _,row in effects.iterrows():
    group=runs[(runs.feed_size==row.feed_size)&(runs.synthetic_fraction==row.synthetic_fraction)&(runs.activity_multiplier==row.activity_multiplier)]
    wide=group.pivot(index='replicate',columns='policy',values=row.metric)
    diff=wide.adaptive-wide[row.control]
    mean=diff.mean();se=diff.std(ddof=1)/np.sqrt(200);half=t.ppf(.975,199)*se
    assert np.allclose([row['mean'],row.ci95_low,row.ci95_high],[mean,mean-half,mean+half],atol=1e-13)
for _,row in interactions.iterrows():
    group=runs[(runs.feed_size==row.feed_size)&(runs.synthetic_fraction==row.synthetic_fraction)]
    wide=group.pivot(index='replicate',columns=['policy','activity_multiplier'],values='exposure_minus_production')
    diff=wide[('adaptive',10.)]-wide[(row.control,10.)]-wide[('adaptive',2.)]+wide[(row.control,2.)]
    half=t.ppf(.975,199)*diff.std(ddof=1)/np.sqrt(200)
    assert np.allclose([row['mean'],row.ci95_low,row.ci95_high],[diff.mean(),diff.mean()-half,diff.mean()+half],atol=1e-13)
# Independent Holm implementation via rank order and cumulative maxima.
for family in [effects[effects.metric=='exposure_minus_production'],interactions]:
    p=family.p_value.to_numpy();indices=np.argsort(p);ranked=p[indices]
    adjusted=np.minimum(1,np.maximum.accumulate(ranked*np.arange(len(p),0,-1)))
    assert np.allclose(family.p_holm.to_numpy()[indices],adjusted,atol=1e-13)
for name in ['synthetic','human']:
    assert (runs[f'{name}_creation_attempts']>=runs[f'{name}_posts']).all()
    assert np.allclose(runs[f'{name}_receipts_per_attempt'],runs[f'{name}_acceptance_rate']*runs[f'{name}_exposures_per_post'],atol=1e-13)
meta=json.loads((directory/'metadata.json').read_text())
assert meta['runs_sha256']==hashlib.sha256((directory/'runs.csv').read_bytes()).hexdigest()
print('PASS: 7,200 unique runs; 36 complete conditions with 200 worlds each')
print('PASS: independently recomputed paired differences, interaction intervals and both Holm families')
print('PASS: attempt/accepted-post accounting and U = acceptance x receipts-per-post identity')
print('PASS: immutable raw data hash matches recorded run metadata')
