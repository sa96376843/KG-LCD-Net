# -*- coding: utf-8 -*-
"""
Figure generation for the KG-LCD-Net paper (all labels in English).
Reads cached probability maps from ../data/cache/*.npz and writes
publication figures into ../ESWA_latex/figs/.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

FIG = Path(__file__).resolve().parent.parent / 'ESWA_latex' / 'figs'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.6,
                     'figure.dpi': 200, 'savefig.bbox': 'tight'})


def overlay_tpfnfp(prob, gt, fov, thr=0.5):
    """ESWA-style colour overlay: TP white, FN red, FP green, on the image."""
    pb = (prob > thr)
    gb = gt > 0
    m = fov > 0
    out = np.zeros((*gt.shape, 3), np.float32)
    out[(pb & gb)] = [1, 1, 1]        # TP  white
    out[(~pb) & gb & m] = [1, 0, 0]   # FN  red
    out[pb & ~gb & m] = [0, 1, 0]     # FP  green
    return out


def fig_qualitative(images, gts, fovs, prob_dict, idxs, names, path):
    """Grid: rows=images, cols=[fundus, GT, method overlays...]."""
    R, C = len(idxs), 2 + len(prob_dict)
    f, ax = plt.subplots(R, C, figsize=(1.55 * C, 1.55 * R))
    headers = ['Fundus', 'Ground truth'] + names
    for c, h in enumerate(headers):
        ax[0, c].set_title(h, fontsize=8)
    for r, i in enumerate(idxs):
        ax[r, 0].imshow(images[i]); ax[r, 1].imshow(gts[i], cmap='gray')
        for c, key in enumerate(prob_dict):
            ax[r, 2 + c].imshow(overlay_tpfnfp(prob_dict[key][i], gts[i], fovs[i]))
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    f.savefig(FIG / path); plt.close(f)


def fig_roc_pr(probs_dict, gts, fovs, path):
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    f, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))
    P = np.concatenate([p[f > 0] for p, f in zip(list(probs_dict.values())[0], fovs)])
    G = np.concatenate([g[f > 0] for g, f in zip(gts, fovs)])
    for name, probs in probs_dict.items():
        P = np.concatenate([p[f > 0] for p, f in zip(probs, fovs)])
        fpr, tpr, _ = roc_curve(G, P)
        prec, rec, _ = precision_recall_curve(G, P)
        ax[0].plot(fpr, tpr, lw=1.2, label=f'{name} (AUC={auc(fpr, tpr):.4f})')
        ax[1].plot(rec, prec, lw=1.2, label=f'{name} (AP={auc(rec, prec):.4f})')
    ax[0].set_xlabel('False positive rate'); ax[0].set_ylabel('True positive rate')
    ax[0].set_title('ROC'); ax[1].set_title('Precision–recall')
    ax[1].set_xlabel('Recall'); ax[1].set_ylabel('Precision')
    for a in ax:
        a.legend(fontsize=7); a.grid(alpha=0.3)
    f.savefig(FIG / path); plt.close(f)


def fig_training_curves(hists, path):
    f, ax = plt.subplots(figsize=(4.6, 2.8))
    for name, h in hists.items():
        h = np.array(h)
        k = max(len(h) // 100, 1)
        sm = np.convolve(h, np.ones(k) / k, 'valid')
        ax.plot(np.arange(len(sm)) * k, sm, lw=1.2, label=name)
    ax.set_xlabel('Training step'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    f.savefig(FIG / path); plt.close(f)
