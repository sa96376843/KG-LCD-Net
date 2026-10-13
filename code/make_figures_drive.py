# -*- coding: utf-8 -*-
"""Generate all DRIVE-based figures for the paper (real experiment outputs)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

OUT = Path(__file__).resolve().parent.parent / 'ESWA_latex' / 'figs'; OUT.mkdir(exist_ok=True)
CACHE = Path(__file__).resolve().parent.parent / 'data' / 'cache'
plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.6,
                     'figure.dpi': 200, 'savefig.bbox': 'tight',
                     'font.family': 'serif'})

# ---------------- data ----------------
root = Path(__file__).resolve().parent.parent / 'data' / 'DRIVE'
te_im, te_gt, te_fov = [], [], []
for gp in sorted((root / 'test' / '1st_manual').glob('*_manual1.gif')):
    stem = gp.name.split('_')[0]
    te_im.append(np.asarray(Image.open(root/'test'/'images'/f'{stem}_test.tif')
                .convert('RGB'), np.float32)/255.)
    te_gt.append((np.asarray(Image.open(gp).convert('L')) > 127).astype(np.float32))
    mp = list((root/'test'/'mask').glob(f'{stem}*mask*'))[0]
    te_fov.append((np.asarray(Image.open(mp).convert('L')) > 127).astype(np.float32))

pb = np.load(CACHE/'drive_backbone_probs.npz')['te']
pf = np.load(CACHE/'drive_full_probs.npy').astype(np.float32)
pb_lcd = np.load(CACHE/'drive_backbone_lcd_probs.npz')['te']
pf_lcd = np.load(CACHE/'drive_full_lcd_probs.npy').astype(np.float32)
PROBS = {'Backbone': pb, 'Full (KG)': pf, 'Full+LCD': pf_lcd}

def overlay(prob, gt, fov, thr=0.5):
    pbm = prob > thr; gbm = gt > 0; m = fov > 0
    out = np.zeros((*gt.shape, 3), np.float32)
    out[pbm & gbm] = [1, 1, 1]
    out[(~pbm) & gbm & m] = [1, 0, 0]
    out[pbm & ~gbm & m] = [0, 0.85, 0]
    return out

# ---------------- Fig: qualitative comparison (DRIVE) ----------------
idxs = [0, 3, 7]
cols = ['Fundus', 'Ground truth', 'Backbone', 'Full (KG)', 'Full+LCD']
f, ax = plt.subplots(len(idxs), len(cols), figsize=(11.5, 5.2))
for c, h in enumerate(cols):
    ax[0, c].set_title(h, fontsize=10)
for r, i in enumerate(idxs):
    ax[r, 0].imshow(te_im[i]); ax[r, 1].imshow(te_gt[i], cmap='gray')
    ax[r, 2].imshow(overlay(pb[i], te_gt[i], te_fov[i]))
    ax[r, 3].imshow(overlay(pf[i], te_gt[i], te_fov[i]))
    ax[r, 4].imshow(overlay(pf_lcd[i], te_gt[i], te_fov[i]))
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
f.savefig(OUT/'fig_qual_drive.pdf'); plt.close(f)
print('fig_qual_drive done', flush=True)

# ---------------- Fig: zoom patches with LCD bridges ----------------
from skimage.graph import route_through_array
from skimage.morphology import skeletonize
from scipy.ndimage import convolve, label

def endpoints(sk):
    nb = convolve(sk.astype(np.uint8), np.ones((3, 3), int), mode='constant') - sk
    return np.argwhere(sk & (nb == 1))

i = 0
prob, gt, fov = pf[i], te_gt[i], te_fov[i]
M = (prob > 0.5) & (fov > 0)
sk = skeletonize(M)
eps = endpoints(sk)
cost = np.exp(-6*prob) + 0.01; cost[fov == 0] = 1e6
used, paths = set(), []
for a, (y, x) in enumerate(eps):
    if a in used: continue
    d = np.hypot(eps[:, 0]-y, eps[:, 1]-x)
    cand = np.where((d > 0) & (d < 45))[0]
    if not len(cand): continue
    b = cand[np.argmin(d[cand])]
    if b in used: continue
    try:
        path, pc = route_through_array(cost, (y, x), tuple(eps[b]), fully_connected=True)
    except Exception:
        continue
    if pc/max(len(path), 1) < 1.9:
        paths.append(np.array(path)); used.add(a); used.add(b)

# choose a zoom window containing bridges
if paths:
    pth = max(paths, key=len)
    cy, cx = pth[:, 0].mean().astype(int), pth[:, 1].mean().astype(int)
else:
    cy, cx = 300, 300
w = 90
y0, x0 = np.clip(cy-w, 0, 584-2*w), np.clip(cx-w, 0, 565-2*w)
sl = np.s_[y0:y0+2*w, x0:x0+2*w]
f, ax = plt.subplots(1, 4, figsize=(11.5, 2.9))
ax[0].imshow(te_im[i][sl]); ax[0].set_title('Fundus (zoom)')
ax[1].imshow(gt[sl], cmap='gray'); ax[1].set_title('Ground truth')
ax[2].imshow(prob[sl], cmap='viridis'); ax[2].set_title('Full (KG) probability')
ax[3].imshow(overlay(prob, gt, fov)[sl])
for pth in paths:
    inwin = ((pth[:, 0] >= y0) & (pth[:, 0] < y0+2*w) &
             (pth[:, 1] >= x0) & (pth[:, 1] < x0+2*w))
    if inwin.sum() > 2:
        ax[3].plot(pth[inwin, 1]-x0, pth[inwin, 0]-y0, 'c.-', ms=2, lw=1)
ax[3].set_title('Full+LCD bridges (cyan)')
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
f.savefig(OUT/'fig_zoom_lcd.pdf'); plt.close(f)
print('fig_zoom_lcd done, bridges:', len(paths), flush=True)

# ---------------- Fig: training curves ----------------
hb = np.load(CACHE/'hist_backbone.npz'); hf = np.load(CACHE/'hist_full.npz')
f, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
for lab, h, c in [('Backbone', hb, 'tab:blue'), ('Full (KG+clDice)', hf, 'tab:red')]:
    l = np.array(h['loss']); k = 50
    sm = np.convolve(l, np.ones(k)/k, 'valid')
    ax[0].plot(np.arange(len(sm))*k, sm, lw=1.3, color=c, label=lab)
    d = np.array(h['dice']); sm2 = np.convolve(d, np.ones(k)/k, 'valid')
    ax[1].plot(np.arange(len(sm2))*k, sm2, lw=1.3, color=c, label=lab)
ax[0].set_xlabel('Training step'); ax[0].set_ylabel('Loss')
ax[1].set_xlabel('Training step'); ax[1].set_ylabel('Patch Dice')
ax[0].set_title('Training loss'); ax[1].set_title('Training Dice')
for a in ax: a.legend(fontsize=8); a.grid(alpha=0.3)
f.savefig(OUT/'fig_curves.pdf'); plt.close(f)
print('fig_curves done', flush=True)

# ---------------- Fig: ROC + PR ----------------
G = np.concatenate([g[f0 > 0] for g, f0 in zip(te_gt, te_fov)])
f, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
for lab, pr in PROBS.items():
    P = np.concatenate([p[f0 > 0] for p, f0 in zip(pr, te_fov)])
    fpr, tpr, _ = roc_curve(G, P)
    prec, rec, _ = precision_recall_curve(G, P)
    ax[0].plot(fpr, tpr, lw=1.2, label=f'{lab} ({auc(fpr, tpr):.4f})')
    ax[1].plot(rec, prec, lw=1.2, label=f'{lab} ({auc(rec, prec):.4f})')
ax[0].set_xlabel('False positive rate'); ax[0].set_ylabel('True positive rate')
ax[0].set_title('ROC curves (DRIVE)'); ax[1].set_title('Precision–recall (DRIVE)')
ax[1].set_xlabel('Recall'); ax[1].set_ylabel('Precision')
for a in ax: a.legend(fontsize=7); a.grid(alpha=0.3)
f.savefig(OUT/'fig_roc.pdf'); plt.close(f)
print('fig_roc done', flush=True)

# ---------------- Fig: ablation bars ----------------
import json
res = json.load(open(CACHE/'results_drive.json'))['DRIVE']
tags = ['Backbone', 'Backbone+LCD', 'Full', 'Full+LCD']
names = ['Backbone', 'Backbone+LCD', 'Full (KG)', 'Full+LCD']
dice = [res[t]['Dice'] for t in tags]
cld = [res[t]['clDice'] for t in tags]
tsen = [res[t]['ThinSen'] for t in tags]
ce = [res[t]['CE'] for t in tags]
x = np.arange(4); wd = 0.26
f, ax = plt.subplots(1, 2, figsize=(7.6, 2.9))
ax[0].bar(x-wd, dice, wd, label='Dice')
ax[0].bar(x, cld, wd, label='clDice')
ax[0].bar(x+wd, tsen, wd, label='Thin-vessel Sen')
ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=7, rotation=12)
ax[0].set_ylim(0.74, 0.85); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3, axis='y')
ax[0].set_title('Region / topology / thin-vessel metrics')
ax[1].bar(x, ce, 0.5, color='tab:purple')
ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=7, rotation=12)
ax[1].set_ylabel('Connectivity error (lower is better)')
ax[1].set_title('Connected-component error'); ax[1].grid(alpha=0.3, axis='y')
for xi, v in zip(x, ce):
    ax[1].text(xi, v+4, f'{v:.0f}', ha='center', fontsize=7)
f.savefig(OUT/'fig_ablation.pdf'); plt.close(f)
print('fig_ablation done', flush=True)

# ---------------- Fig: multi-layer graph illustration ----------------
from skimage.segmentation import slic, mark_boundaries
i = 0
im, prob, fov = te_im[i], pf[i], te_fov[i]
from skimage.filters import frangi
v = frangi(im[..., 1], sigmas=range(1, 7), black_ridges=False)
v = (v-v.min())/(v.max()-v.min()+1e-8)
seg = slic(im, n_segments=800, compactness=10, mask=fov > 0,
           start_label=0, channel_axis=-1)
from skimage.segmentation import relabel_sequential
seg = seg - seg.min()          # shift mask-outside label (-1) to 0
seg, _, _ = relabel_sequential(seg)
f, ax = plt.subplots(1, 4, figsize=(11.5, 2.9))
ax[0].imshow(mark_boundaries(im, seg, color=(1, 1, 0), mode='thin'))
ax[0].set_title('SLIC superpixels')
# layer edge strength visualisation: aggregate edges per layer
def layer_map(wfun, title, axx):
    n = seg.max()+1
    W = np.zeros(seg.shape)
    # mean weight of boundary edges per superpixel
    from scipy.ndimage import find_objects
    U, Vl, Wt = [], [], []
    for a_, b_ in ((seg[:, :-1], seg[:, 1:]), (seg[:-1, :], seg[1:, :])):
        m = a_ != b_
        U += list(a_[m].ravel()); Vl += list(b_[m].ravel())
        Wt += list(wfun(a_[m].ravel(), b_[m].ravel()))
    U = np.array(U); Vl = np.array(Vl); Wt = np.array(Wt)
    acc = np.zeros(n); cnt = np.zeros(n)
    np.add.at(acc, U, Wt); np.add.at(cnt, U, 1)
    np.add.at(acc, Vl, Wt); np.add.at(cnt, Vl, 1)
    cnt[cnt == 0] = 1
    meanw = acc/cnt
    axx.imshow(meanw[seg], cmap='inferno'); axx.set_title(title)
cnt = np.bincount(seg.ravel(), minlength=seg.max()+1).clip(min=1)
imc = [np.bincount(seg.ravel(), weights=im[..., c].ravel())/cnt for c in range(3)]
pc = np.bincount(seg.ravel(), weights=prob.ravel())/cnt
vc = np.bincount(seg.ravel(), weights=v.ravel())/cnt
yy, xx = np.indices(seg.shape)
cyc = np.bincount(seg.ravel(), weights=yy.ravel())/cnt
cxc = np.bincount(seg.ravel(), weights=xx.ravel())/cnt
C3 = np.stack(imc, 1)
layer_map(lambda a, b: np.exp(-(((C3[a]-C3[b])**2).sum(1))/0.02),
          'Layer 1: appearance', ax[1])
layer_map(lambda a, b: np.exp(-(((cyc[a]-cyc[b])**2+(cxc[a]-cxc[b])**2))/(2*15**2)),
          'Layer 2: spatial', ax[2])
layer_map(lambda a, b: np.exp(-((pc[a]-pc[b])**2+(vc[a]-vc[b])**2)/0.05),
          'Layer 3: semantic', ax[3])
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
f.savefig(OUT/'fig_multilayer.pdf'); plt.close(f)
print('fig_multilayer done', flush=True)
