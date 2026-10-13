# -*- coding: utf-8 -*-
"""Retrain DRIVE full model (KG + clDice) with the correct KG adjacency,
using the final deliverable architecture in model.py."""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from skimage.filters import frangi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import KGLCDUNet, count_params
from losses import dice_loss, cldice_loss

OUT = Path(__file__).resolve().parent.parent / 'data' / 'cache'
TMP = Path(__file__).resolve().parent.parent / 'data' / 'cache' / 'chase_arrays'; TMP.mkdir(exist_ok=True)

root = Path(__file__).resolve().parent.parent / 'data' / 'DRIVE'

def load(kind, gt_pat):
    ims, gts, fovs = [], [], []
    for gp in sorted((root/kind/'1st_manual').glob(gt_pat)):
        stem = gp.name.split('_')[0]
        ims.append(np.asarray(Image.open(root/kind/'images'/f'{stem}_{kind}.tif')
                   .convert('RGB'), np.float32)/255.)
        gts.append((np.asarray(Image.open(gp).convert('L')) > 127).astype(np.float32))
        mp = list((root/kind/'mask').glob(f'{stem}*mask*'))[0]
        fovs.append((np.asarray(Image.open(mp).convert('L')) > 127).astype(np.float32))
    return ims, gts, fovs

tr_im, tr_gt, tr_fov = load('training', '*.gif')
te_im, te_gt, te_fov = load('test', '*_manual1.gif')

def vess(ims):
    out = []
    for i, im in enumerate(ims):
        v = frangi(im[..., 1], sigmas=range(1, 7), black_ridges=False)
        v = (v-v.min())/(v.max()-v.min()+1e-8)
        out.append(v.astype(np.float32))
        print('vesselness', i, flush=True)
    return out

if (TMP/'drive_tr_v.npy').exists():
    tr_v = list(np.load(TMP/'drive_tr_v.npy'))
    te_v = list(np.load(TMP/'drive_te_v.npy'))
else:
    tr_v = [x.astype(np.float16) for x in vess(tr_im)]; te_v = [x.astype(np.float16) for x in vess(te_im)]
    np.save(TMP/'drive_tr_v.npy', np.stack(tr_v))
    np.save(TMP/'drive_te_v.npy', np.stack(te_v))
print('data ready', flush=True)

def sample(bs, size=96):
    Xs, Ys = [], []
    for _ in range(bs):
        i = np.random.randint(len(tr_im))
        img, g, f0, vv = tr_im[i], tr_gt[i], tr_fov[i], tr_v[i]
        H, W = g.shape
        yy, xx = np.where(f0 > 0); k = np.random.randint(len(yy))
        cy, cx = yy[k], xx[k]; s = size
        y0 = int(np.clip(cy-s//2+np.random.randint(-s//3, s//3), 0, H-s))
        x0 = int(np.clip(cx-s//2+np.random.randint(-s//3, s//3), 0, W-s))
        sl = np.s_[y0:y0+s, x0:x0+s]
        pim, pgt, pv = img[sl].copy(), g[sl].copy(), vv[sl].copy()
        if np.random.rand() < .5: pim, pgt, pv = pim[:, ::-1], pgt[:, ::-1], pv[:, ::-1]
        if np.random.rand() < .5: pim, pgt, pv = pim[::-1], pgt[::-1], pv[::-1]
        r = np.random.randint(4)
        pim, pgt, pv = np.rot90(pim, r).copy(), np.rot90(pgt, r).copy(), np.rot90(pv, r).copy()
        if np.random.rand() < .5: pim = np.clip(pim, 1e-4, 1)**np.random.uniform(.7, 1.4)
        x = np.concatenate([pim, pv[..., None]], -1)
        Xs.append(torch.from_numpy(x).permute(2, 0, 1).float())
        Ys.append(torch.from_numpy(pgt)[None].float())
    return torch.stack(Xs), torch.stack(Ys)

torch.manual_seed(7)
m = KGLCDUNet(cin=4, base=24, use_kg=True)
print('params', count_params(m), flush=True)
opt = torch.optim.Adam(m.parameters(), lr=3e-4)
total = 2500
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-5)
hist = []
m.train()
t0 = time.time()
for step in range(total):
    x, y = sample(8)
    logit = m(x); p = torch.sigmoid(logit)
    loss = 0.5*F.binary_cross_entropy_with_logits(logit, y) + dice_loss(p, y) \
           + 0.5*cldice_loss(p, y)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    hist.append(float(loss.detach()))
    if (step+1) % 250 == 0:
        print(f'step {step+1}/{total} loss {hist[-1]:.4f} t={time.time()-t0:.0f}s',
              flush=True)

@torch.no_grad()
def probs(m):
    m.eval(); out = []
    for img, vv in zip(te_im, te_v):
        H, W = vv.shape
        ph, pw = (16-H % 16) % 16, (16-W % 16) % 16
        x = np.concatenate([img, vv[..., None]], -1)
        x = np.pad(x, ((0, ph), (0, pw), (0, 0)))
        t = torch.from_numpy(x).permute(2, 0, 1)[None].float()
        p = torch.sigmoid(m(t))[0, 0].numpy()
        out.append(p[:H, :W])
    return np.stack(out)

pf = probs(m)
np.save(TMP/'drive_full_probs.npy', pf.astype(np.float16))
torch.save(m.state_dict(), OUT/'drive_full.pt')
np.save(TMP/'drive_hist_full.npy', np.array(hist))
print('DONE full drive', time.time()-t0, flush=True)
