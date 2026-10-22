"""Rebuild the sensitivity figure and numeric LaTeX table from saved summaries."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    data = Path('results/feed_capacity_robustness')
    summary = pd.read_csv(data/'summary.csv')
    paired = pd.read_csv(data/'paired_activity_effects.csv')
    assert len(summary) == 18 and (summary.n == 200).all()
    assert len(paired) == 6 and (paired.n == 200).all()
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.1), sharey=True)
    for ax, fraction in zip(axes, [.05, .25]):
        for feed, color, marker in [(1, '#1764a0', 'o'), (5, '#b45b16', 's'), (20, '#25816b', '^')]:
            rows = summary[(summary.synthetic_fraction == fraction) & (summary.feed_size == feed)]
            rows = rows.sort_values('activity_multiplier')
            ax.errorbar(rows.activity_multiplier, rows['mean'],
                        yerr=np.array([rows['mean']-rows.ci95_low, rows.ci95_high-rows['mean']]),
                        color=color, marker=marker, capsize=3, linewidth=1.5,
                        label=f'Feed capacity {feed}')
        ax.axhline(0, color='#555555', linewidth=.8, linestyle='--')
        ax.set_xscale('log')
        ax.set_xticks([1, 3, 15], ['1×', '3×', '15×'])
        ax.minorticks_off()
        ax.set_title(f'{fraction:.0%} producer prevalence')
        ax.set_xlabel('Relative activity')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Mean relative exposure D')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=[0, .11, 1, 1])
    output = Path('report/imgs/robustness')
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output/'feed_capacity.png', dpi=250)
    fig.savefig(output/'feed_capacity.pdf')
    plt.close(fig)
    lines = [r'\begin{table}[!htbp]', r'\caption{Feed-capacity sensitivity across 200 paired worlds. '
             r'Condition means and paired changes in $D$ use the same worlds. '
             r'Intervals are pointwise 95\% $t$ intervals, without multiplicity adjustment.}',
             r'\label{tab:feed-capacity}', r'\centering\small',
             r'\begin{tabular}{rrrrrl}', r'\hline',
             r'Prevalence & $\alpha$ & Mean $D_3$ & Mean $D_{15}$ & Mean change & 95\% CI \\', r'\hline']
    for _, row in paired.sort_values(['synthetic_fraction', 'feed_size']).iterrows():
        group = summary[(summary.feed_size == row.feed_size) &
                        (summary.synthetic_fraction == row.synthetic_fraction)].set_index('activity_multiplier')
        lines.append(f"{row.synthetic_fraction:.0%}".replace('%', r'\%') +
                     f" & {int(row.feed_size)} & {group.loc[3., 'mean']:.4f} & {group.loc[15., 'mean']:.4f}"
                     f" & {row['mean']:.4f} & [{row.ci95_low:.4f}, {row.ci95_high:.4f}]" + r' \\')
    lines += [r'\hline', r'\end{tabular}', r'\end{table}']
    Path('report/feed_capacity_table.tex').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
