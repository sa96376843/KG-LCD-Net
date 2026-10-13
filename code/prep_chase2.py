# -*- coding: utf-8 -*-
"""Prepare CHASE_DB1 -> /tmp (local disk), compact dtypes."""
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.filters import frangi
from skimage.morphology import binary_closing, disk, remove_small_objects

TMP = Path(__file__).resolve().parent.parent / 'data' / 'cache' / 'chase_arrays'; TMP.mkdir(exist_ok=True)
chroot = Path(__file__).resolve().parent.parent / 'data' / 'CHASEDB1'
jpgs = sorted(chroot.glob('Image_*.jpg'))
print('found', len(jpgs), flush=True)

gt_u = np.stack([(np.asarray(Image.open(chroot / f'{p.stem}_1stHO.png')
                 .convert('L')) > 127) for p in jpgs]).astype(np.uint8)
np.save(TMP/'ch_gt.npy', gt_u)
print('gt saved', gt_u.shape, flush=True)

fovs, vs = [], []
for i, p in enumerate(jpgs):
    im = np.asarray(Image.open(p).convert('RGB'), np.float32) / 255.
    fov = binary_closing(im.mean(2) > 18/255., disk(5))
    fovs.append(remove_small_objects(fov, 20000))
    v = frangi(im[..., 1], sigmas=range(1, 7), black_ridges=False)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    vs.append(v.astype(np.float16))
    print('fov+vesselness', i, flush=True)
np.save(TMP/'ch_fov.npy', np.stack(fovs).astype(np.uint8))
np.save(TMP/'ch_v.npy', np.stack(vs))
print('fov+v saved', flush=True)

# images as uint8
im_u = np.stack([np.asarray(Image.open(p).convert('RGB'))
                 for p in jpgs]).astype(np.uint8)
np.save(TMP/'ch_im.npy', im_u)
print('im saved; all done', flush=True)
