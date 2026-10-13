# -*- coding: utf-8 -*-
"""
End-to-end experiment runner: trains backbone / full KG-LCD-Net on DRIVE and
CHASE_DB1, applies ML-LCD refinement, computes all metrics and dumps
results.json + cached probability maps used by figures.py / tables.py.

Usage (CPU, ~4 h total):  python run_experiments.py
"""
import json
import numpy as np
import torch
from pathlib import Path

from dataset_prep import load_drive, load_chase, vesselness_channel
from train import (PatchDS, make_run, train_chunk, full_image_probs,
                   metrics_from_probs, hard_cldice)
from lcd import lcd_reconnect

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'data' / 'cache'
CACHE.mkdir(exist_ok=True)


def train_and_eval(name, use_kg, lam_cld, ds, te_ims, te_gts, te_fovs,
                   total_steps=3500, chunk=250):
    run = make_run(use_kg, lam_cld, ds, total_steps)
    while run['step'] < total_steps:
        train_chunk(run, min(chunk, total_steps - run['step']))
        print(f"{name}: {run['step']}/{total_steps}", flush=True)
    torch.save(run['model'].state_dict(), CACHE / f'{name}.pt')
    probs = full_image_probs(run['model'], te_ims,
                             vess_cache['te'] if 'drive' in name.lower()
                             else vess_cache['ch_te'])
    np.savez(CACHE / f'{name}_probs.npz', te=probs)
    return run, probs


if __name__ == '__main__':
    # ---------------- DRIVE ----------------
    (tr_im, tr_gt, tr_fov), (te_im, te_gt, te_fov) = load_drive(
        ROOT / 'data' / 'DRIVE')
    tr_v = vesselness_channel(tr_im)
    te_v = vesselness_channel(te_im)
    vess_cache = {'te': te_v}
    ds = PatchDS(tr_im, tr_gt, tr_fov, tr_v)

    results = {}
    rb, pb = train_and_eval('drive_backbone', False, 0.0, ds,
                            te_im, te_gt, te_fov)
    rf, pf = train_and_eval('drive_full', True, 0.5, ds,
                            te_im, te_gt, te_fov)
    # ML-LCD refinement
    pb_lcd = np.stack([lcd_reconnect(pb[i], te_fov[i], radius=45,
                                     path_thr=1.9, min_comp=0,
                                     boost_val=0.52)[0]
                       for i in range(len(pb))])
    pf_lcd = np.stack([lcd_reconnect(pf[i], te_fov[i], radius=45,
                                     path_thr=1.9, min_comp=0,
                                     boost_val=0.52)[0]
                       for i in range(len(pf))])
    for tag, pr in [('Backbone', pb), ('Backbone+LCD', pb_lcd),
                    ('Full', pf), ('Full+LCD', pf_lcd)]:
        m = metrics_from_probs(pr, te_gt, te_fov)
        c, _, _ = hard_cldice(pr, te_gt, te_fov)
        m['clDice'] = c
        results.setdefault('DRIVE', {})[tag] = m
        print('DRIVE', tag, m, flush=True)

    # ---------------- CHASE_DB1 ----------------
    (ci_tr, cg_tr, cf_tr), (ci_te, cg_te, cf_te) = load_chase(
        ROOT / 'data' / 'CHASEDB1')
    cv_tr = vesselness_channel(ci_tr)
    cv_te = vesselness_channel(ci_te)
    vess_cache['ch_te'] = cv_te
    dsc = PatchDS(ci_tr, cg_tr, cf_tr, cv_tr)
    rcb, pcb = train_and_eval('chase_backbone', False, 0.0, dsc,
                              ci_te, cg_te, cf_te)
    rcf, pcf = train_and_eval('chase_full', True, 0.5, dsc,
                              ci_te, cg_te, cf_te)
    for tag, pr in [('Backbone', pcb), ('Full', pcf)]:
        m = metrics_from_probs(pr, cg_te, cf_te)
        c, _, _ = hard_cldice(pr, cg_te, cf_te)
        m['clDice'] = c
        results.setdefault('CHASE', {})[tag] = m
        print('CHASE', tag, m, flush=True)

    (CACHE / 'results.json').write_text(json.dumps(results, indent=2))
    print('saved', CACHE / 'results.json')
