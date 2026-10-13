# -*- coding: utf-8 -*-
"""Train backbone + full KG-LCD-Net on CHASE_DB1 (first 20 train / last 8 test),
save checkpoints and test probability maps. CPU script."""
import sys, time, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import KGLCDUNet          # deliverable architecture (clean version)
from losses import dice_loss, cldice_loss

TMP = Path(__file__).resolve().parent.parent / 'data' / 'cache' / 'chase_arrays'
OUT = Path(__file__).resolve().parent.parent / 'data' / 'cache'

im = np.load(TMP/'ch_im.npy').astype(np.float32)/255.
gt = np.load(TMP/'ch_gt.npy'); fov = np.load(TMP/'ch_fov.npy'); v = np.load(TMP/'ch_v.npy')
TR = list(range(0, 20)); TE = list(range(20, 28))
tr_im, tr_gt, tr_fov, tr_v = [im[i] for i in TR], [gt[i] for i in TR], [fov[i] for i in TR], [v[i] for i in TR]
te_im, te_gt, te_fov, te_v = [im[i] for i in TE], [gt[i] for i in TE], [fov[i] for i in TE], [v[i] for i in TE]
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


def train(use_kg, lam, total=2500, lr=3e-4):
    torch.manual_seed(42 if not use_kg else 7)
    m = KGLCDUNet(cin=4, base=24, use_kg=use_kg)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-5)
    hist = []
    m.train()
    for step in range(total):
        x, y = sample(8)
        logit = m(x); p = torch.sigmoid(logit)
        loss = 0.5*F.binary_cross_entropy_with_logits(logit, y) + dice_loss(p, y)
        if lam > 0: loss = loss + lam*cldice_loss(p, y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        hist.append(float(loss.detach()))
        if (step+1) % 250 == 0:
            print(f'  kg={use_kg} step {step+1}/{total} loss {hist[-1]:.4f}', flush=True)
    return m, hist


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


t0 = time.time()
mb_, hb = train(False, 0.0)
pb = probs(mb_)
np.save(TMP/'chase_backbone_probs.npy', pb)
torch.save(mb_.state_dict(), OUT/'chase_backbone.pt')
np.save(TMP/'chase_hist_b.npy', np.array(hb))
print('backbone done', time.time()-t0, flush=True)

mf_, hf = train(True, 0.5)
pf = probs(mf_)
np.save(TMP/'chase_full_probs.npy', pf)
torch.save(mf_.state_dict(), OUT/'chase_full.pt')
np.save(TMP/'chase_hist_f.npy', np.array(hf))
print('full done', time.time()-t0, flush=True)
