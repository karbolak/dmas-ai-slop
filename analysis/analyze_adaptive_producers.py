"""Validate and summarize paired producer policies; emit report-ready evidence.

D contrasts are Holm-adjusted across all 24 policy/condition comparisons.
The eight high-minus-moderate activity interactions form a separate family.
Diagnostic intervals and all displayed confidence intervals are pointwise.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis.analyze_volume_sweep_final import holm_adjust

METRICS=['exposure_minus_production','synthetic_production_share','synthetic_exposure_share',
         'synthetic_exposures_per_post','synthetic_receipts_per_attempt','synthetic_acceptance_rate',
         'synthetic_post_receipts_zero_fraction','concentration_change',
         'early_mean_max_strategy_fraction','late_mean_max_strategy_fraction']
GROUP=['feed_size','synthetic_fraction','activity_multiplier']


def estimate(values):
    a=np.asarray(values,dtype=float)
    if len(a)<2 or not np.isfinite(a).all():
        raise ValueError('Need at least two finite observations in every paired estimate')
    mean=a.mean(); sd=a.std(ddof=1); se=sd/np.sqrt(len(a))
    half=t.ppf(.975,len(a)-1)*se
    p=2*t.sf(abs(mean/se),len(a)-1) if se else float(mean==0)
    return dict(n=len(a),mean=mean,ci95_low=mean-half,ci95_high=mean+half,
                p_value=p,dz=mean/sd if sd else np.nan)


def paired_estimate(frame,metric,left,right):
    wide=frame.pivot(index='replicate',columns='policy',values=metric)
    if left not in wide or right not in wide or wide[[left,right]].isna().any().any():
        raise ValueError('Missing within-world policy pairs')
    return estimate(wide[left]-wide[right])



def verify_recorded_hashes(hashes):
    """Audit captured files if available, without requiring the run machine."""
    checked=0; unavailable=[]
    for name,digest in hashes.items():
        path=Path(name)
        if not path.is_file():
            unavailable.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Recorded source/binary fingerprint differs: {name}')
        else:
            checked+=1
    return checked,unavailable


def validate(frame,directory):
    cfg=json.loads((directory/'config.json').read_text())
    grid=[]
    for block in cfg['design']:
        grid.extend(itertools.product(block['feed_sizes'],block['fractions'],
                                      block['activity_multipliers'],cfg['producer']['policies']))
    expected=set(itertools.product(range(cfg['seeds']['replicates']),grid))
    keys=['replicate']+GROUP+['policy']
    observed={(row[0],tuple(row[1:])) for row in frame[keys].itertuples(index=False,name=None)}
    if observed!=expected or frame.duplicated(keys).any():
        raise ValueError('Incomplete or duplicated experimental design')
    assert (frame.total_activations==cfg['simulation']['iterations']+1).all()
    assert np.isfinite(frame[METRICS]).all().all()
    assert (frame.synthetic_creation_attempts>=frame.synthetic_posts).all()
    manifest=pd.read_csv(directory/'seed_manifest.csv').set_index('replicate')
    for name in ['network_seed','opinion_seed','producer_assignment_seed','simulation_seed']:
        assert (frame[name]==frame.replicate.map(manifest[name])).all()
    assert np.allclose(frame.exposure_minus_production,
        frame.synthetic_exposures/(frame.synthetic_exposures+frame.human_exposures)
        -frame.synthetic_posts/(frame.synthetic_posts+frame.human_posts),atol=1e-14)
    for name,column in [('accepted','synthetic_posts'),('receipts','synthetic_exposures'),('attempts','synthetic_creation_attempts')]:
        assert (frame[[f'synthetic_{name}_arm_{a}' for a in range(cfg['producer']['n_strategies'])]].sum(axis=1)==frame[column]).all()
    meta=json.loads((directory/'metadata.json').read_text())
    digest=hashlib.sha256((directory/'runs.csv').read_bytes()).hexdigest()
    assert digest==meta['runs_sha256'] and meta['runs']==len(frame)
    checked,unavailable=verify_recorded_hashes(meta['hashes'])
    return dict(worlds=cfg['seeds']['replicates'],conditions=len(grid),runs=len(frame),
                unique_keys=True,seed_pairing=True,attempt_acceptance_receipts_reconciled=True,
                finite_outcomes=True,source_and_data_hashes=not unavailable,
                source_hashes_checked=checked,source_hashes_unavailable=unavailable)


def analyze(frame):
    summary=[]; effects=[]; interactions=[]
    for key,group in frame.groupby(GROUP+['policy']):
        for metric in METRICS:
            summary.append(dict(zip(GROUP+['policy'],key),metric=metric,**estimate(group[metric])))
    for key,group in frame.groupby(GROUP):
        for control in ['baseline','random']:
            for metric in METRICS:
                effects.append(dict(zip(GROUP,key),control=control,metric=metric,
                                    **paired_estimate(group,metric,'adaptive',control)))
    effects=pd.DataFrame(effects)
    mask=effects.metric=='exposure_minus_production'
    effects.loc[mask,'p_holm']=holm_adjust(effects.loc[mask,'p_value'].tolist())
    for key,group in frame.groupby(GROUP[:2]):
        wide=group.pivot(index='replicate',columns=['policy','activity_multiplier'],values='exposure_minus_production')
        for control in ['baseline','random']:
            diff=(wide[('adaptive',10.)]-wide[(control,10.)])-(wide[('adaptive',2.)]-wide[(control,2.)])
            interactions.append(dict(zip(GROUP[:2],key),control=control,**estimate(diff)))
    interactions=pd.DataFrame(interactions)
    interactions['p_holm']=holm_adjust(interactions.p_value.tolist())
    return pd.DataFrame(summary),effects,interactions


def make_figures(summary,effects,report,frame):
    path=report/'imgs/adaptive';path.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    styles={'baseline':('#1764a0','o'),'random':('#8a6c00','s'),'adaptive':('#a52848','^')}
    def points(ax,rows,label,color,marker):
        rows=rows.sort_values('activity_multiplier')
        ax.errorbar(rows.activity_multiplier,rows['mean'],
            yerr=np.array([rows['mean']-rows.ci95_low,rows.ci95_high-rows['mean']]),
            label=label,color=color,marker=marker,capsize=3,linewidth=1.4)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5),sharey=True)
    for ax,fraction in zip(axes,[.05,.1]):
        for policy,(color,marker) in styles.items():
            rows=summary[(summary.feed_size==1)&(summary.synthetic_fraction==fraction)&(summary.policy==policy)&(summary.metric=='exposure_minus_production')]
            points(ax,rows,policy.capitalize(),color,marker)
        ax.axhline(0,color='#666',ls='--',lw=.8);ax.set_xticks([1,2,5,10])
        ax.set_xlabel('Relative activity');ax.set_title(f'{fraction:.0%} producer prevalence');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Mean relative exposure D')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=[0,.1,1,1]);fig.savefig(path/'policy_comparison.pdf');fig.savefig(path/'policy_comparison.png',dpi=250);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5),sharey=True)
    for ax,multiplier in zip(axes,[2.,10.]):
        for control,(color,marker) in list(styles.items())[:2]:
            rows=effects[(effects.synthetic_fraction==.05)&(effects.activity_multiplier==multiplier)&(effects.control==control)&(effects.metric=='exposure_minus_production')].sort_values('feed_size')
            ax.errorbar(rows.feed_size,rows['mean'],yerr=np.array([rows['mean']-rows.ci95_low,rows.ci95_high-rows['mean']]),
                        marker=marker,color=color,label=f'Adaptive − {control}',capsize=3)
        ax.set_title(f'{multiplier:g}× activity');ax.set_xticks([1,3,5]);ax.set_xlabel('Feed capacity');ax.axhline(0,color='#666',ls='--',lw=.8);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Paired change in D')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=[0,.1,1,1]);fig.savefig(path/'adaptation_capacity.pdf');fig.savefig(path/'adaptation_capacity.png',dpi=250);plt.close(fig)


    fig,axes=plt.subplots(1,2,figsize=(7.2,3.8),sharey=True)
    labels=[f'{policy.capitalize()} {phase}' for policy in ['baseline','random','adaptive'] for phase in ['early','late']]
    for ax,fraction in zip(axes,[.05,.1]):
        selected=frame[(frame.feed_size==1)&(frame.synthetic_fraction==fraction)&(frame.activity_multiplier==10)]
        matrix=[]
        for policy in ['baseline','random','adaptive']:
            group=selected[selected.policy==policy]
            for phase in ['early','late']:
                counts=group[[f'synthetic_{phase}_attempts_arm_{a}' for a in range(5)]].sum().to_numpy()
                matrix.append(counts/counts.sum())
        matrix=np.asarray(matrix)
        im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=.4,aspect='auto')
        for r in range(6):
            for c in range(5): ax.text(c,r,f'{matrix[r,c]:.2f}',ha='center',va='center',fontsize=11,color='white' if matrix[r,c]>.25 else 'black')
        ax.set_xticks(range(5),['0','1','2','3','4']);ax.set_xlabel('Opinion bin');ax.set_yticks(range(6),labels)
        ax.set_title(f'{fraction:.0%} prevalence')
    fig.tight_layout(rect=[0,0,.94,1]);cax=fig.add_axes([.95,.2,.015,.6]);fig.colorbar(im,cax=cax,label='Share of innovation attempts')
    fig.savefig(path/'strategy_choices.pdf',bbox_inches='tight');fig.savefig(path/'strategy_choices.png',dpi=250,bbox_inches='tight');plt.close(fig)


def make_tables(effects,interactions,report):
    def pformat(p): return '$<0.001$' if p<.001 else f'{p:.3f}'
    def ci(row): return f"[{row.ci95_low:.4f}, {row.ci95_high:.4f}]"
    primary=effects[(effects.feed_size==1)&(effects.metric=='exposure_minus_production')]
    lines=[r'\begin{table}[!htbp]',r'\caption{Paired effect of content adaptation with a one-item feed (200 worlds). Changes are adaptive minus the stated control. Intervals are pointwise 95\% $t$ intervals; $p$ values use Holm adjustment across all 24 policy contrasts, including larger feeds.}',r'\label{tab:adaptive-main}',r'\centering\small\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{rrlrrl}',r'\hline',r'Prevalence & Activity & Control & Mean $\Delta D$ & 95\% CI & Adjusted $p$ \\',r'\hline']
    for _,row in primary.sort_values(['synthetic_fraction','activity_multiplier','control']).iterrows():
        lines.append(f"{100*row.synthetic_fraction:g}"+r'\%'+f" & {row.activity_multiplier:g}"+r'$\times$'+f" & {row.control} & {row['mean']:.4f} & {ci(row)} & {pformat(row.p_holm)}"+r' \\')
    lines += [r'\hline',r'\end{tabular}',r'\end{table}']
    (report/'adaptive_primary_table.tex').write_text('\n'.join(lines)+'\n')
    rows=effects[(effects.feed_size==1)&(effects.control=='baseline')&effects.activity_multiplier.isin([2.,10.])]
    labels=[('synthetic_acceptance_rate',r'$\Delta A_s$'),('synthetic_exposures_per_post',r'$\Delta E_s$'),('synthetic_receipts_per_attempt',r'$\Delta U_s$'),('concentration_change',r'$\Delta C$')]
    lines=[r'\begin{table}[!htbp]',r'\caption{Mechanism diagnostics: adaptive minus baseline, with a one-item feed. $A_s$ is posting acceptance, $E_s$ is receipts per accepted post, $U_s$ is receipts per attempt, and $C$ is the late-minus-early change in mean per-producer maximum strategy share. These are descriptive paired mean differences.}',r'\label{tab:adaptive-diagnostics}',r'\centering\small',r'\begin{tabular}{rrrrrr}',r'\hline',r'Prevalence & Activity & '+ ' & '.join(label for metric,label in labels)+r' \\',r'\hline']
    for (fraction,multiplier),group in rows.groupby(['synthetic_fraction','activity_multiplier']):
        v=group.set_index('metric')['mean']
        lines.append(f'{100*fraction:g}'+r'\%'+f' & {multiplier:g}'+r'$\times$'+' & '+' & '.join(f'{v[metric]:.4f}' for metric,label in labels)+r' \\')
    lines += [r'\hline',r'\end{tabular}',r'\end{table}']
    (report/'adaptive_diagnostics_table.tex').write_text('\n'.join(lines)+'\n')
    lines=[r'\begin{table}[!htbp]',r'\caption{Does the adaptation effect change from 2$\times$ to 10$\times$ activity? Each estimate is $(D_{\rm adaptive}-D_{\rm baseline})_{10}-(D_{\rm adaptive}-D_{\rm baseline})_2$, paired within world. Intervals are pointwise; Holm correction covers all eight interaction contrasts, including the random control.}',r'\label{tab:adaptive-interaction}',r'\centering\small',r'\begin{tabular}{rrrrl}',r'\hline',r'Prevalence & Feed & Mean interaction & 95\% CI & Adjusted $p$ \\',r'\hline']
    for _,row in interactions[interactions.control=='baseline'].sort_values(['feed_size','synthetic_fraction']).iterrows():
        lines.append(f'{100*row.synthetic_fraction:g}'+r'\%'+f" & {int(row.feed_size)} & {row['mean']:.4f} & {ci(row)} & {pformat(row.p_holm)}"+r' \\')
    lines += [r'\hline',r'\end{tabular}',r'\end{table}']
    (report/'adaptive_interaction_table.tex').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--directory',type=Path,default=Path('results/adaptive_producers'))
    args=parser.parse_args();directory=args.directory
    frame=pd.read_csv(directory/'runs.csv');validation=validate(frame,directory)
    out=directory/'analysis';out.mkdir(exist_ok=True)
    summary,effects,interactions=analyze(frame)
    for name,data in [('summary',summary),('paired_policy_effects',effects),('activity_interactions',interactions)]: data.to_csv(out/f'{name}.csv',index=False)
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    report=Path('report');make_figures(summary,effects,report,frame);make_tables(effects,interactions,report)
    (out/'analysis_metadata.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runs_sha256=hashlib.sha256((directory/'runs.csv').read_bytes()).hexdigest(),D_contrast_family=24,interaction_family=8),indent=2)+'\n')
    print(json.dumps(validation,indent=2))
    print(effects[(effects.feed_size==1)&(effects.metric=='exposure_minus_production')].to_string(index=False))
    print(interactions.to_string(index=False))


if __name__=='__main__': main()
