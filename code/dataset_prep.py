# -*- coding: utf-8 -*-
"""
Dataset preparation for DRIVE and CHASE_DB1.

Expected layout (see README):
  data/DRIVE/training/images/*.tif, data/DRIVE/training/1st_manual/*.gif,
  data/DRIVE/training/mask/*.gif
  data/DRIVE/test/images/*.tif, data/DRIVE/test/1st_manual/*.png,
  data/DRIVE/test/mask/*.png
  data/CHASEDB1/Image_*.jpg + Image_*_1stHO.png
Outputs float arrays + Frangi vesselness channel; CHASE FOVs are generated.
"""
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.filters import frangi
from skimage.morphology import binary_closing, disk, remove_small_objects


def load_drive(root):
    root = Path(root)
    def split(kind, ext_img, ext_gt, gt_dir='1st_manual'):
        ims, gts, fovs = [], [], []
        img_dir = root / kind / 'images'
        gt_paths = sorted((root / kind / gt_dir).glob(f'*.{ext_gt}'))
        for gp in gt_paths:
            stem = gp.name.split('_')[0]
            ip = img_dir / f'{stem}_{kind}.{ext_img}'
            if not ip.exists():
                ip = img_dir / f'{stem}.{ext_img}'
            mp = root / kind / 'mask' / f'{stem}_mask.gif'
            if not mp.exists():
                mp = root / kind / 'mask' / f'{stem}_mask.png'
            im = np.asarray(Image.open(ip).convert('RGB'), np.float32) / 255.
            gt = (np.asarray(Image.open(gp).convert('L')) > 127).astype(np.float32)
            fov = (np.asarray(Image.open(mp).convert('L')) > 127).astype(np.float32)
            ims.append(im); gts.append(gt); fovs.append(fov)
        return ims, gts, fovs
    tr = split('training', 'tif', 'gif')
    te = split('test', 'tif', 'png')
    return tr, te


def load_chase(root, n_train=20):
    root = Path(root)
    ims, gts, fovs = [], [], []
    for ip in sorted(root.glob('Image_*.jpg')):
        stem = ip.stem
        gp = root / f'{stem}_1stHO.png'
        im = np.asarray(Image.open(ip).convert('RGB'), np.float32) / 255.
        gt = (np.asarray(Image.open(gp).convert('L')) > 127).astype(np.float32)
        gray = im.mean(2)
        fov = binary_closing(gray > 18 / 255., disk(5))
        fov = remove_small_objects(fov, 20000).astype(np.float32)
        ims.append(im); gts.append(gt); fovs.append(fov)
    return (ims[:n_train], gts[:n_train], fovs[:n_train]), \
           (ims[n_train:], gts[n_train:], fovs[n_train:])


def vesselness_channel(ims):
    """Frangi vesselness of the green channel, normalised to [0,1]."""
    out = []
    for im in ims:
        g = im[..., 1]
        v = frangi(g, sigmas=range(1, 7), black_ridges=False)
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
        out.append(v.astype(np.float32))
    return out


if __name__ == '__main__':
    tr, te = load_drive('data/DRIVE')
    print('DRIVE train/test:', len(tr[0]), len(te[0]), tr[0][0].shape)
    (ci, cg, cf), _ = load_chase('data/CHASEDB1')
    print('CHASE train:', len(ci), ci[0].shape)
