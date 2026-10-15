"""Plot measured recipe results; refuses incomplete sweeps."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--sweep', required=True)
ap.add_argument('--quality', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
d = json.loads(Path(a.sweep).read_text())
quality = json.loads(Path(a.quality).read_text())
quality_ok = quality['top1_agreement'] >= .99 and quality['mean_kl_top20'] <= .001
assert d['status'] == 'DONE', 'Only complete measured sweeps can be plotted'
bg, panel, text, muted, green, cyan = '#06110d', '#0d1d16', '#e9f4ee', '#9db9ac', '#adff69', '#52cee3'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': text,
    'axes.labelcolor': muted, 'xtick.color': muted, 'ytick.color': text,
    'axes.edgecolor': panel, 'savefig.facecolor': bg})
fig, (ax, pre) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1.2, 1]})
fig.patch.set_facecolor(bg)
fig.subplots_adjust(left=.08, right=.97, top=.72, bottom=.23, wspace=.28)
fig.text(.08, .94, 'FIELD NOTES / RTX 5090 / NEMOTRON 3.5 SUPER', color=green, fontsize=10, family='monospace')
fig.text(.08, .865, 'More conversations, same GPU', fontsize=25, weight='bold')
fig.text(.08, .81, 'EXL3 3-bit HQ · CPU expert offloading · natural-end completions', color=muted)
for axis in (ax, pre):
    axis.set_facecolor(panel); axis.spines[['top', 'right', 'left']].set_visible(False)
    axis.grid(axis='x', color='#263b30', linewidth=.8); axis.set_axisbelow(True)
    axis.tick_params(axis='y', length=0, pad=10)
ys = np.arange(4)
for offset, key, color, label in [(-.16, 'aggregate', green, 'Total throughput'),
                                 (.16, 'per_stream_mean', cyan, 'Per conversation')]:
    rows = [d['decode'][f'C{c}'][key] for c in range(1, 5)]
    vals = [r['median'] for r in rows]
    ax.barh(ys+offset, vals, height=.25, color=color, alpha=.35, label=label)
    for y, row in zip(ys+offset, rows):
        ax.hlines(y, row['min'], row['max'], color=color, linewidth=1.7)
        ax.scatter(row['all'], [y]*len(row['all']), color=color, edgecolors=bg, linewidth=.6, s=24, zorder=4)
        ax.vlines(row['median'], y-.105, y+.105, color=color, linewidth=2.5, zorder=5)
        ax.text(row['max']+1, y, f"{row['median']:.1f}", va='center', color=color, fontsize=10)
ax.set_yticks(ys, ['1 user', '2 users', '3 users', '4 users']); ax.invert_yaxis()
ax.set_xlim(0, max(d['decode'][f'C{c}']['aggregate']['max'] for c in range(1, 5))*1.22)
ax.set_xlabel('output tokens / second', labelpad=13)
ax.set_title('Generation speed', loc='left', color=text, fontsize=15, pad=15)
ax.legend(loc='upper left', bbox_to_anchor=(0, -.21), ncol=2, frameon=False, labelcolor=text, fontsize=9)
lengths = [8192, 16384, 32768, 65536]
for i, n in enumerate(lengths):
    row = d['prefill'][str(n)]
    pre.barh(i, row['median'], color=green, alpha=.35, height=.48)
    pre.hlines(i, row['min'], row['max'], color=green, linewidth=1.5)
    pre.scatter(row['all'], [i]*len(row['all']), color=green, edgecolors=bg, linewidth=.6, s=24, zorder=4)
    pre.vlines(row['median'], i-.2, i+.2, color=green, linewidth=2.5, zorder=5)
    pre.text(row['max']+30, i, f"{row['median']:,.0f}", color=green, va='center', fontsize=10)
pre.set_yticks(range(4), ['8K input', '16K input', '32K input', '64K input']); pre.invert_yaxis()
pre.set_xlim(0, max(d['prefill'][str(n)]['max'] for n in lengths)*1.25)
pre.set_xlabel('input tokens / time to first token', labelpad=13)
pre.set_title('Prompt processing', loc='left', color=text, fontsize=15, pad=15)
long = d['decode']['C1@32k']['aggregate']['median']
if not quality_ok:
    fig.text(.08, .14, f"Diagnostic run · runtime parity gate FAILED ({quality['top1_agreement']:.1%} token agreement; 99% required)", color='#ffc27b', fontsize=10)
fig.text(.08, .085, f'32K prompt, one user: {long:.1f} tok/s   ·   128K configured context   ·   thinking off', color=muted, fontsize=10)
fig.text(.08, .035, 'Bars + ticks: medians. Dots: every run. Decode n=2; prefill n=3.  acrogenesis', color=muted, fontsize=9)
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
for suffix in ('.png', '.svg'):
    fig.savefig(out.with_suffix(suffix), dpi=160)
