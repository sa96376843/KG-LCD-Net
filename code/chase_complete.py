"""Frozen CHASE_DB1 run. Train and evaluate in separate processes."""
import argparse
import gc
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).absolute().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import torch
import torch.nn.functional as F

from model import KGLCDUNet
from losses import dice_loss, cldice_loss

CACHE = ROOT / "data" / "cache" / "chase_arrays"
OUT = ROOT / "data" / "cache" / "chase_completion"
FILES = ("ch_im.npy", "ch_gt.npy", "ch_fov.npy", "ch_v.npy")
STEPS = 2500
BATCH = 8
SEED = 42
MAX_SECONDS = 6 * 3600
MIN_FREE = 200 * 1024 * 1024


def preflight():
    from PIL import Image
    shapes = {}
    for name in FILES:
        path = CACHE / name
        if not path.is_file():
            raise FileNotFoundError(path)
        # Do not use mmap_mode: the available CUDA environment cannot resolve
        # memory-mapped files in this Windows workspace.
        array = np.load(path)
        shapes[name] = {"shape": list(array.shape), "dtype": str(array.dtype)}
        del array
    assert shapes["ch_im.npy"]["shape"] == [28, 960, 999, 3]
    assert all(shapes[n]["shape"] == [28, 960, 999]
               for n in ("ch_gt.npy", "ch_fov.npy", "ch_v.npy"))
    images = sorted((ROOT / "data" / "CHASEDB1").glob("Image_*.jpg"))
    if len(images) != 28 or any(not p.with_name(p.stem + "_1stHO.png").is_file()
                                for p in images):
        raise RuntimeError("CHASE_DB1 image/first-observer label pairs are incomplete")
    cached_im = np.load(CACHE / "ch_im.npy")
    cached_gt = np.load(CACHE / "ch_gt.npy")
    rgb_deltas = []
    for i, path in enumerate(images):
        raw_im = np.asarray(Image.open(path).convert("RGB"))
        raw_gt = np.asarray(Image.open(path.with_name(path.stem + "_1stHO.png"))
                            .convert("L")) > 127
        delta = np.abs(cached_im[i].astype(np.int16) - raw_im.astype(np.int16))
        rgb_deltas.append({"name": path.name, "mean_abs": float(delta.mean()),
                           "max_abs": int(delta.max())})
        # The cache was decoded on Linux; Windows Pillow/libjpeg differs by
        # small RGB rounding values. This tolerance excludes a different image.
        if delta.mean() > .5 or delta.max() > 64:
            raise RuntimeError(f"Cached RGB image differs materially from {path.name}")
        if not np.array_equal(cached_gt[i] > 0, raw_gt):
            raise RuntimeError(f"Cached label differs from {path.name}")
    del cached_im, cached_gt
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen protocol requires the available CUDA GPU")
    if importlib.util.find_spec("skimage") is None:
        raise RuntimeError("scikit-image is missing; install into this run's vendor directory")
    from skimage.morphology import skeletonize  # noqa: F401
    from skimage.graph import route_through_array  # noqa: F401
    from sklearn.metrics import roc_auc_score  # noqa: F401
    free = shutil.disk_usage(HERE).free
    if free < MIN_FREE:
        raise RuntimeError(f"Low disk space: {free} bytes free")
    return {"arrays": shapes, "rgb_decoder_deltas": rgb_deltas,
            "train": [p.name for p in images[:20]],
            "test": [p.name for p in images[20:]], "torch": torch.__version__,
            "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0),
            "disk_free_bytes": free}


def load_arrays():
    im, gt, fov, vess = (np.load(CACHE / n) for n in FILES)
    coords = [np.flatnonzero(fov[i]) for i in range(20)]
    if any(len(c) == 0 for c in coords):
        raise RuntimeError("Empty training FOV")
    return im, gt, fov, vess, coords


def sample_batch(arrays):
    im, gt, fov, vess, coords = arrays
    xs, ys = [], []
    for _ in range(BATCH):
        i = np.random.randint(20)
        k = coords[i][np.random.randint(len(coords[i]))]
        cy, cx = divmod(int(k), 999)
        y0 = int(np.clip(cy - 48 + np.random.randint(-32, 32), 0, 960 - 96))
        x0 = int(np.clip(cx - 48 + np.random.randint(-32, 32), 0, 999 - 96))
        sl = np.s_[y0:y0 + 96, x0:x0 + 96]
        rgb = im[i][sl].astype(np.float32) / 255.
        label = gt[i][sl].astype(np.float32)
        v = vess[i][sl].astype(np.float32)
        if np.random.rand() < .5:
            rgb, label, v = rgb[:, ::-1], label[:, ::-1], v[:, ::-1]
        if np.random.rand() < .5:
            rgb, label, v = rgb[::-1], label[::-1], v[::-1]
        r = np.random.randint(4)
        rgb, label, v = np.rot90(rgb, r), np.rot90(label, r), np.rot90(v, r)
        if np.random.rand() < .5:
            rgb = np.clip(rgb, 1e-4, 1) ** np.random.uniform(.7, 1.4)
        x = np.concatenate([rgb, v[..., None]], -1).copy()
        xs.append(torch.from_numpy(x).permute(2, 0, 1))
        ys.append(torch.from_numpy(label.copy())[None])
    return torch.stack(xs).float().cuda(), torch.stack(ys).float().cuda()


def save_state(path, step, model, opt, sched, history):
    state = {"step": step, "model": model.state_dict(), "optimizer": opt.state_dict(),
             "scheduler": sched.state_dict(), "history": history,
             "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all()}
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def log_event(message):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {message}"
    print(line, flush=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def train_one(name, use_kg, lam, arrays, deadline):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = KGLCDUNet(cin=4, base=24, use_kg=use_kg).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=1e-5)
    ckpt = OUT / f"{name}_latest.pt"
    final = OUT / f"{name}_final.pt"
    step, history = 0, []
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cuda", weights_only=False)
        step, history = state["step"], state["history"]
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
    model.train()
    for s in range(step, STEPS):
        if time.monotonic() >= deadline or shutil.disk_usage(HERE).free < MIN_FREE:
            save_state(ckpt, s, model, opt, sched, history)
            raise RuntimeError(f"Stopped safely at {name} step {s}: time/disk budget")
        x, y = sample_batch(arrays)
        logit = model(x)
        p = torch.sigmoid(logit)
        loss = .5 * F.binary_cross_entropy_with_logits(logit, y) + dice_loss(p, y)
        if lam:
            loss = loss + lam * cldice_loss(p, y)
        if not bool(torch.isfinite(loss)):
            save_state(ckpt, s, model, opt, sched, history)
            raise RuntimeError(f"Nonfinite loss at {name} step {s + 1}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        history.append(float(loss.detach().cpu()))
        if (s + 1) % 500 == 0:
            save_state(ckpt, s + 1, model, opt, sched, history)
            log_event(f"{name}: {s + 1}/{STEPS}, loss={history[-1]:.6f}")
    torch.save(model.cpu().state_dict(), final)
    np.save(OUT / f"{name}_loss.npy", np.asarray(history, dtype=np.float64))
    log_event(f"{name}: training complete")
    return model


@torch.no_grad()
def predict(model, arrays):
    im, _, _, vess, _ = arrays
    model.eval().cpu()
    out = []
    for i in range(20, 28):
        rgb = im[i].astype(np.float32) / 255.
        v = vess[i].astype(np.float32)
        x = np.concatenate([rgb, v[..., None]], -1)
        h, w = v.shape
        x = np.pad(x, ((0, (-h) % 16), (0, (-w) % 16), (0, 0)))
        t = torch.from_numpy(x).permute(2, 0, 1)[None].float()
        out.append(torch.sigmoid(model(t))[0, 0, :h, :w].numpy())
    return np.stack(out).astype(np.float32)


def measure(probs, gt, fov):
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    from sklearn.metrics import roc_auc_score
    rows = []
    for p, g, f in zip(probs, gt[20:], fov[20:]):
        m = f > 0
        gb = (g > 0) & m
        pb = (p > .5) & m
        tp, fp = int((pb & gb).sum()), int((pb & ~gb).sum())
        fn, tn = int((~pb & gb).sum()), int((~pb & ~gb & m).sum())
        sp, sg = skeletonize(pb), skeletonize(gb)
        prec = (sp & gb).sum() / max(int(sp.sum()), 1)
        sens = (sg & pb).sum() / max(int(sg.sum()), 1)
        row = {"Dice": 2*tp / max(2*tp+fp+fn, 1),
               "IoU": tp / max(tp+fp+fn, 1),
               "Sen": tp / max(tp+fn, 1), "Spe": tn / max(tn+fp, 1),
               "Acc": (tp+tn) / max(tp+tn+fp+fn, 1),
               "AUC": float(roc_auc_score(gb[m], p[m])),
               "clDice": 2*prec*sens / max(prec+sens, 1e-9),
               "ConnErr": abs(ndi.label(pb)[1] - ndi.label(gb)[1])}
        rows.append({k: float(v) for k, v in row.items()})
    return rows, {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if sum((args.preflight, args.train_only, args.evaluate_only)) != 1:
        parser.error("Select exactly one mode")
    info = preflight()
    if args.preflight:
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
        return
    OUT.mkdir(exist_ok=True)
    (OUT / "preflight.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if args.train_only:
        log_event("Frozen CHASE_DB1 training phase started")
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        arrays = load_arrays()
        deadline = time.monotonic() + MAX_SECONDS
        for name, use_kg, lam in (("backbone", False, 0.), ("full", True, .5)):
            if (OUT / f"{name}_final.pt").is_file() and (OUT / f"{name}_probs.npy").is_file():
                log_event(f"{name}: completed checkpoint and probability map found; skipped")
                continue
            model = train_one(name, use_kg, lam, arrays, deadline)
            prob = predict(model, arrays)
            np.save(OUT / f"{name}_probs.npy", prob)
            log_event(f"{name}: full-image predictions saved")
            del model, prob
            gc.collect()
        log_event("Both CHASE_DB1 models and original probability maps completed")
        return
    from lcd import lcd_reconnect
    log_event("Frozen CHASE_DB1 evaluation phase started")
    gt = np.load(CACHE / "ch_gt.npy")
    fov = np.load(CACHE / "ch_fov.npy")
    results = {}
    for name in ("backbone", "full"):
        prob_path = OUT / f"{name}_probs.npy"
        if not (OUT / f"{name}_final.pt").is_file() or not prob_path.is_file():
            raise RuntimeError(f"{name} training outputs are incomplete")
        prob = np.load(prob_path)
        rows, mean = measure(prob, gt, fov)
        results[name] = {"per_image": rows, "mean": mean}
        bridge = np.empty_like(prob)
        for i in range(8):
            bridge[i], _, _ = lcd_reconnect(prob[i], fov[20+i],
                                             thr=.5, radius=45, path_thr=1.9,
                                             min_comp=0, boost_val=.52)
            log_event(f"{name}: bridged test image {i+1}/8")
            gc.collect()
        np.save(OUT / f"{name}_bridge_probs.npy", bridge)
        rows, mean = measure(bridge, gt, fov)
        results[name + "_bridge"] = {"per_image": rows, "mean": mean}
        (OUT / "results_partial.json").write_text(json.dumps(results, indent=2),
                                                     encoding="utf-8")
        del prob, bridge
        gc.collect()
    (OUT / "results_chase.json").write_text(json.dumps(results, indent=2),
                                              encoding="utf-8")
    log_event("All four CHASE_DB1 conditions completed")


if __name__ == "__main__":
    main()
