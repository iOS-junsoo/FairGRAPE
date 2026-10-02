#!/usr/bin/env python3
"""Compare original continuous OFF with restarted cap smoke runs, using saved CSVs."""
import argparse
import csv
from pathlib import Path

from plot_dogscats_alpha07_layer_survival import collect, draw, read_csv, ITERATIONS
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path,
                        default=repo / 'dogscats_diagnostics/layer_cap_smoke_20260920_062004')
    args = parser.parse_args()
    data, records = collect(repo, args.run_dir)
    records = [r for r in records if r['cap'] != 'OFF']
    historical = [r for r in read_csv(repo / 'docs/dogscats_alpha09_alpha07_analysis_20260920/layers.csv')
                  if r['alpha'] == '0.7' and int(r['iter']) in ITERATIONS]
    indexed = {(int(r['iter']), int(r['block'].split('.')[1])): r for r in historical}
    assert len(historical) == len(indexed) == 68
    data['off'] = {}
    previous = {}
    for iteration in ITERATIONS:
        values = []
        for layer in range(1, 18):
            row = indexed[(iteration, layer)]
            original = int(row['original_channels'])
            before, removed, alive = (int(row[k]) for k in ('active_before', 'removed_now', 'active_after'))
            assert before - removed == alive and 0 <= alive <= before <= original
            if iteration > 25:
                assert before == previous[layer]
            percent = 100.0 * alive / original
            values.append(percent)
            records.append(dict(alpha=0.7, cap='OFF', iteration=iteration, layer=layer,
                                original_channels=original, surviving_channels=alive,
                                survival_percent=percent, source='original continuous run'))
            previous[layer] = alive
        data['off'][iteration] = values
    assert data['off'][25] == data['cap10'][25] == data['cap20'][25]
    assert [data['off'][it][0] for it in ITERATIONS] == [28.125, 25.0, 12.5, 9.375]
    assert len(records) == 204
    cap_order = {'OFF': 0, '10%': 1, '20%': 2}
    records.sort(key=lambda r: (cap_order[r['cap']], r['iteration'], r['layer']))
    out = args.run_dir / 'layer_survival_alpha07_historical_off'
    out.mkdir(exist_ok=True)
    with (out / 'layer_survival_percent.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    conditions = (('off', 'OFF (original continuous run)'),
                  ('cap10', '10% (restarted smoke)'),
                  ('cap20', '20% (restarted smoke)'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    note = ('Survival = remaining / original channels in each layer.  '
            'Iteration 25 is shared; caps begin at iteration 26.')
    comparison_note = 'OFF: original continuous run.  10% / 20%: separate runs restarted from the iteration-25 checkpoint.'
    with PdfPages(out / 'alpha07_layer_survival.pdf') as pdf:
        for condition, label in conditions:
            fig, ax = plt.subplots(figsize=(16, 5.5))
            draw(ax, data, condition, label)
            fig.subplots_adjust(left=0.075, right=0.988, bottom=0.23, top=0.79)
            fig.text(0.075, 0.045, note + '\n' + comparison_note,
                     fontsize=10, color='#475569', linespacing=1.65)
            fig.savefig(out / ('alpha07_{}_survival.png'.format(condition)), dpi=150)
            fig.savefig(out / ('alpha07_{}_survival.svg'.format(condition)))
            pdf.savefig(fig)
            plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(16, 15))
    for ax, (condition, label) in zip(axes, conditions):
        draw(ax, data, condition, label)
    fig.subplots_adjust(left=0.075, right=0.988, bottom=0.11, top=0.94, hspace=0.62)
    fig.text(0.075, 0.024, note + '\n' + comparison_note,
             fontsize=11, color='#475569', linespacing=1.7)
    fig.savefig(out / 'alpha07_all_caps_survival.png', dpi=150)
    plt.close(fig)
    print('Verified 204 data points; original OFF layer 1: 9 -> 8 -> 4 -> 3 channels.')
    print('10% and 20% data unchanged; shared iteration-25 channel counts verified.')
    print(out)


if __name__ == '__main__':
    main()
