#!/usr/bin/env python3
"""Plot saved alpha=.7 channel masks; performs no training or inference."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np


ITERATIONS = (25, 26, 27, 28)
CONDITIONS = (('off', 'OFF'), ('cap10', '10%'), ('cap20', '20%'))
COLORS = ('#94A3B8', '#2563EB', '#0D9488', '#EA580C')


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def collect(repo, run):
    baseline_rows = [r for r in read_csv(repo / 'docs/dogscats_alpha09_alpha07_analysis_20260920/layers.csv')
                     if r['alpha'] == '0.7' and int(r['iter']) == 25]
    baseline = {int(r['block'].split('.')[1]): r for r in baseline_rows}
    assert len(baseline_rows) == len(baseline) == 17
    assert set(baseline) == set(range(1, 18))
    source = read_csv(run / 'layers_comparison.csv')
    data, records = {}, []
    for condition, label in CONDITIONS:
        data[condition] = {}
        rows = [r for r in source if r['arm'] == 'alpha0p7_' + condition]
        indexed = {(int(r['iteration']), int(r['block'].split('.')[1])): r for r in rows}
        assert len(rows) == len(indexed) == 51
        previous = {layer: int(row['active_after']) for layer, row in baseline.items()}
        for iteration in ITERATIONS:
            values = []
            for layer in range(1, 18):
                original = int(baseline[layer]['original_channels'])
                if iteration == 25:
                    alive = int(baseline[layer]['active_after'])
                    origin = 'shared historical iteration-25 checkpoint'
                else:
                    row = indexed[(iteration, layer)]
                    assert int(row['original']) == original
                    assert int(row['active']) == previous[layer]
                    alive = int(row['active_after'])
                    assert previous[layer] - int(row['removed']) == alive
                    assert 0 <= alive <= previous[layer]
                    origin = 'new matched smoke run'
                assert 0 <= alive <= original
                percent = 100.0 * alive / original
                values.append(percent)
                records.append(dict(alpha=0.7, cap=label, iteration=iteration, layer=layer,
                                    original_channels=original, surviving_channels=alive,
                                    survival_percent=percent, source=origin))
                previous[layer] = alive
            data[condition][iteration] = values
    assert all(data['off'][it] == data['cap20'][it] for it in ITERATIONS)
    return data, records


def draw(ax, data, condition, label):
    x = np.arange(1, 18)
    width = 0.19
    for index, (iteration, color) in enumerate(zip(ITERATIONS, COLORS)):
        legend = 'Iteration 25 (shared start)' if iteration == 25 else 'Iteration {}'.format(iteration)
        ax.bar(x + (index - 1.5) * width, data[condition][iteration], width=width,
               label=legend, color=color, edgecolor='white', linewidth=0.45, zorder=3)
    ax.set_title('Alpha = 0.7  |  Removal cap: {}'.format(label), loc='left',
                 fontsize=17, fontweight='bold', pad=42, color='#0F172A')
    ax.legend(ncol=4, loc='lower left', bbox_to_anchor=(0, 1.015), frameon=False,
              borderaxespad=0, handlelength=1.5, columnspacing=2.4, fontsize=11)
    ax.set_xlim(0.4, 17.6)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel('Layer number (features.1 to features.17)', labelpad=9)
    ax.set_ylabel('Surviving channels (% of original)', labelpad=10)
    ax.grid(axis='y', color='#E2E8F0', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for name in ('left', 'bottom'):
        ax.spines[name].set_color('#CBD5E1')
    ax.tick_params(axis='both', colors='#334155', length=0, pad=7)


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path,
                        default=repo / 'dogscats_diagnostics/layer_cap_smoke_20260920_062004')
    args = parser.parse_args()
    out = args.run_dir / 'layer_survival_alpha07'
    out.mkdir(exist_ok=True)
    data, records = collect(repo, args.run_dir)
    with (out / 'layer_survival_percent.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    note = ('Survival = channels remaining after the iteration / original channels in that layer.  '
            'Caps begin at iteration 26.')
    with PdfPages(out / 'alpha07_layer_survival.pdf') as pdf:
        for condition, label in CONDITIONS:
            fig, ax = plt.subplots(figsize=(16, 5.3))
            draw(ax, data, condition, label)
            fig.subplots_adjust(left=0.075, right=0.988, bottom=0.19, top=0.78)
            fig.text(0.075, 0.035, note, fontsize=10, color='#475569')
            fig.savefig(out / ('alpha07_{}_survival.png'.format(condition)), dpi=150)
            fig.savefig(out / ('alpha07_{}_survival.svg'.format(condition)))
            pdf.savefig(fig)
            plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(16, 15))
    for ax, (condition, label) in zip(axes, CONDITIONS):
        draw(ax, data, condition, label)
    fig.subplots_adjust(left=0.075, right=0.988, bottom=0.11, top=0.94, hspace=0.62)
    fig.text(0.075, 0.024, note + '\nOFF and 20% have identical channel counts at all plotted iterations.',
             fontsize=11, color='#475569', linespacing=1.7)
    fig.savefig(out / 'alpha07_all_caps_survival.png', dpi=150)
    plt.close(fig)
    print('Verified {} points: 3 caps x 4 iterations x 17 layers.'.format(len(records)))
    print('Shared checkpoint alignment, channel accounting, and non-regrowth checks passed.')
    for path in sorted(out.iterdir()):
        print(path)


if __name__ == '__main__':
    main()
