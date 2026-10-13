# -*- coding: utf-8 -*-
"""Training and evaluation utilities for KG-LCD-Net."""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from skimage.morphology import skeletonize

from model import KGLCDUNet
from losses import combined_loss, dice_loss, cldice_loss


# --------------------------------------------------------------------------- #
class PatchDS(Dataset):
    """Random 96x96 FOV-centred patches with flips/rot90/gamma augmentation."""
    def __init__(self, ims, gts, fovs, vess, size=96, epoch_len=1000):
        self.ims, self.gts, self.fovs, self.vess = ims, gts, fovs, vess
        self.size, self.epoch_len = size, epoch_len

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, _):
        i = np.random.randint(len(self.ims))
        im, gt, fov, v = self.ims[i], self.gts[i], self.fovs[i], self.vess[i]
        H, W = gt.shape
        ys, xs = np.where(fov > 0)
        k = np.random.randint(len(ys))
        cy, cx = ys[k], xs[k]
        s = self.size
        y0 = np.clip(cy - s // 2 + np.random.randint(-s // 3, s // 3), 0, H - s)
        x0 = np.clip(cx - s // 2 + np.random.randint(-s // 3, s // 3), 0, W - s)
        sl = np.s_[y0:y0 + s, x0:x0 + s]
        pim, pgt, pv = im[sl], gt[sl], v[sl]
        if np.random.rand() < 0.5:
            pim, pgt, pv = pim[:, ::-1], pgt[:, ::-1], pv[:, ::-1]
        if np.random.rand() < 0.5:
            pim, pgt, pv = pim[::-1], pgt[::-1], pv[::-1]
        r = np.random.randint(4)
        pim, pgt, pv = (np.rot90(a, r) for a in (pim, pgt, pv))
        if np.random.rand() < 0.5:
            pim = np.clip(pim, 1e-4, 1) ** np.random.uniform(0.7, 1.4)
        x = np.concatenate([pim, pv[..., None]], -1)
        return (torch.from_numpy(x.copy()).permute(2, 0, 1).float(),
                torch.from_numpy(pgt.copy())[None].float())


# --------------------------------------------------------------------------- #
def make_run(use_kg, lam_cld, ds, total_steps, bs=8, lr=3e-4):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = KGLCDUNet(cin=4, base=24, use_kg=use_kg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True,
                                     num_workers=0, drop_last=True)
    return dict(model=model, opt=opt, dl=iter(dl), lam_cld=lam_cld,
                total=total_steps, step=0, hist=[], dev=dev)


def train_chunk(run, n_steps):
    """Train for n_steps with L = 0.5*BCE + Dice + lambda*clDice.

    NOTE: dice/clDice losses are computed on sigmoid probabilities
    (combined_loss handles this correctly)."""
    model, opt, lam = run['model'], run['opt'], run['lam_cld']
    model.train()
    for _ in range(n_steps):
        try:
            x, y = next(run['dl'])
        except StopIteration:
            run['dl'] = iter(torch.utils.data.DataLoader(
                run['dl'].dataset, batch_size=8, shuffle=True))
            x, y = next(run['dl'])
        x, y = x.to(run['dev']), y.to(run['dev'])
        loss = combined_loss(model(x), y, lam)
        opt.zero_grad(); loss.backward(); opt.step()
        run['hist'].append(float(loss))
        run['step'] += 1


# --------------------------------------------------------------------------- #
@torch.no_grad()
def full_image_probs(model, ims, vess, dev='cpu'):
    model.eval()
    out = []
    for im, v in zip(ims, vess):
        H, W = v.shape
        ph, pw = (16 - H % 16) % 16, (16 - W % 16) % 16
        x = np.concatenate([im, v[..., None]], -1)
        x = np.pad(x, ((0, ph), (0, pw), (0, 0)))
        t = torch.from_numpy(x).permute(2, 0, 1)[None].float().to(dev)
        p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
        out.append(p[:H, :W])
    return np.stack(out)


# --------------------------------------------------------------------------- #
def metrics_from_probs(probs, gts, fovs, thr=0.5):
    """Dice / IoU / Sen / Spe / Acc / AUC, computed within the FOV."""
    dice, iou, sen, spe, acc, auc = [], [], [], [], [], []
    for p, g, f in zip(probs, gts, fovs):
        m = f > 0
        pb = (p > thr) & m
        gb, gm = g > 0, g[m]
        tp = (pb & gb).sum(); fp = (pb & ~gb).sum()
        fn = (~pb & gb).sum(); tn = (~pb & ~gb & m).sum()
        dice.append(2 * tp / max(2 * tp + fp + fn, 1))
        iou.append(tp / max(tp + fp + fn, 1))
        sen.append(tp / max(tp + fn, 1))
        spe.append(tn / max(tn + fp, 1))
        acc.append((tp + tn) / max(tp + tn + fp + fn, 1))
        auc.append(roc_auc_score(gm, p[m]))
    return {k: float(np.mean(v)) for k, v in
            dict(Dice=dice, IoU=iou, Sen=sen, Spe=spe, Acc=acc, AUC=auc).items()}


def hard_cldice(probs, gts, fovs, thr=0.5):
    """Hard clDice on skeletonised thresholded predictions."""
    vals, precs, senss = [], [], []
    for p, g, f in zip(probs, gts, fovs):
        m = f > 0
        pb = (p > thr) & m
        gb = (g > 0) & m
        sp, sg = skeletonize(pb), skeletonize(gb)
        tprec = (sp & gb).sum() / max(sp.sum(), 1)
        tsens = (sg & pb).sum() / max(sg.sum(), 1)
        c = 2 * tprec * tsens / max(tprec + tsens, 1e-9)
        vals.append(c); precs.append(tprec); senss.append(tsens)
    return float(np.mean(vals)), float(np.mean(precs)), float(np.mean(senss))
