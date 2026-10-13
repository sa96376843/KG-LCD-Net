# -*- coding: utf-8 -*-
"""Regenerate all LaTeX tables for the KG-LCD-Net paper.
Reads cached metrics from ../data/cache/results.json (produced by
run_experiments.py) and writes .tex files into ../ESWA_latex/tables/."""
import json
from pathlib import Path

TAB = Path(__file__).resolve().parent.parent / 'ESWA_latex' / 'tables'
TAB.mkdir(exist_ok=True)


def tex_table(caption, label, header, rows, col_fmt=None, note=None):
    col_fmt = col_fmt or 'l' + 'c' * (len(header) - 1)
    s = ['\\begin{table}[!htbp]', '\\centering', '\\small',
         f'\\caption{{{caption}}}', f'\\label{{{label}}}',
         f'\\begin{{tabular}}{{{col_fmt}}}', '\\toprule',
         ' & '.join(header) + ' \\\\', '\\midrule']
    for r in rows:
        s.append(' & '.join(str(x) for x in r) + ' \\\\')
    s += ['\\bottomrule', '\\end{tabular}']
    if note:
        s.append(f'\\\\[2pt]\\footnotesize {note}')
    s.append('\\end{table}')
    out = '\n'.join(s)
    (TAB / f'{label.replace("tab:", "")}.tex').write_text(out)
    return out


def table_datasets():
    header = ['Dataset', 'Images', 'Resolution', 'Train/Test', 'Field of view', 'Labels']
    rows = [
        ['DRIVE', '40', '565$\\times$584', '20 / 20', '45$^{\\circ}$', 'Two observers (1st used)'],
        ['CHASE\\_DB1', '28', '999$\\times$960', '20 / 8', '30$^{\\circ}$', 'Two observers (1st used)'],
    ]
    return tex_table('Statistics of the two public retinal fundus datasets used in this study.',
                     'tab:datasets', header, rows)


def table_sota(results, dataset='DRIVE'):
    """SOTA comparison: rows = published methods + ours (real measured)."""
    pub = {
        'DRIVE': [
            ['U-Net \\cite{ronneberger2015unet}', 0.8142, '-', 0.7537, 0.9820, 0.9531, 0.9755],
            ['R2U-Net \\cite{alom2018r2unet}', 0.8171, '-', 0.7792, 0.9813, 0.9556, 0.9784],
            ['LadderNet \\cite{zhuang2018laddernet}', 0.8202, '-', 0.7856, 0.9810, 0.9561, 0.9793],
            ['IterNet \\cite{li2020iternet}', 0.8205, '-', 0.7735, 0.9838, 0.9573, 0.9816],
            ['SA-UNet \\cite{guo2021saunet}', 0.8263, '-', 0.8212, 0.9840, 0.9698, 0.9864],
            ['SGL \\cite{zhou2021sgl}', 0.8316, '-', 0.8380, 0.9834, 0.9705, 0.9886],
            ['MTPA-UNet \\cite{jiang2022mtpa}', 0.8318, '-', 0.8410, '-', 0.9718, '-'],
            ['EFDG-UNet \\cite{zhou2023dualpath}', 0.8412, '-', '-', '-', '-', 0.9886],
        ],
        'CHASE': [
            ['SA-UNet \\cite{guo2021saunet}', '-', '-', 0.8573, '-', 0.9755, 0.9905],
            ['MTPA-UNet \\cite{jiang2022mtpa}', 0.8164, '-', 0.8437, '-', 0.9762, '-'],
            ['TP-UNet \\cite{liu2023tpunet}', 0.8162, '-', 0.8242, '-', 0.9664, '-'],
            ['MR-UNet \\cite{zhou2023dualpath}', 0.8460, '-', '-', 0.9930, 0.9778, 0.9925],
        ],
    }
    header = ['Method', 'Dice', 'clDice', 'Sen', 'Spe', 'Acc', 'AUC']
    rows = pub[dataset]
    ours = results[dataset]['Full+LCD']
    rows.append(['\\textbf{KG-LCD-Net (ours)}'] +
                [f"\\textbf{{{ours[k]:.4f}}}" for k in
                 ['Dice', 'clDice', 'Sen', 'Spe', 'Acc', 'AUC']])
    cap = (f'Comparison with state-of-the-art methods on the {dataset} test set. '
           'Best results of our method in bold; published numbers are taken from the '
           'original papers; ``--\'\' denotes unreported.')
    return tex_table(cap, f'tab:sota_{dataset.lower()}', header, rows)


if __name__ == '__main__':
    print(table_datasets())
