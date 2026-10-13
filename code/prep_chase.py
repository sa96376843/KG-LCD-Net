# -*- coding: utf-8 -*-
"""Prepare CHASE_DB1 arrays (images, GT, FOV, Frangi vesselness) -> cache."""
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.filters import frangi
from skimage.morphology import binary_closing, disk, remove_small_objects

CACHE = Path(__file__).resolve().parent.parent / 'data' / 'cache' / 'chase_arrays'
chroot = Path(__file__).resolve().parent.parent / 'data' / 'CHASEDB1'
jpgs = sorted(chroot.glob('Image_*.jpg'))
print('found', len(jpgs), flush=True)

gt_f = np.stack([(np.asarray(Image.open(chroot / f'{p.stem}_1stHO.png')
                  .convert('L')) > 127).astype(np.float32) for p in jpgs])
np.save(CACHE / 'ch_gt.npy', gt_f)
print('gt saved', flush=True)

fov_f = []
for p in jpgs:
    im = np.asarray(Image.open(p).convert('RGB'), np.float32) / 255.
    fov = binary_closing(im.mean(2) > 18 / 255., disk(5))
    fov_f.append(remove_small_objects(fov, 20000).astype(np.float32))
np.save(CACHE / 'ch_fov.npy', np.stack(fov_f))
print('fov saved', flush=True)

v_f = []
for i, p in enumerate(jpgs):
    im = np.asarray(Image.open(p).convert('RGB'), np.float32) / 255.
    v = frangi(im[..., 1], sigmas=range(1, 7), black_ridges=False)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    v_f.append(v.astype(np.float32))
    print('vesselness', i, flush=True)
np.save(CACHE / 'ch_v.npy', np.stack(v_f))
print('v saved', flush=True)

# images stored as uint8 to save space
im_u = np.stack([(np.asarray(Image.open(p).convert('RGB')))
                 for p in jpgs]).astype(np.uint8)
np.save(CACHE / 'ch_im.npy', im_u)
print('im saved; all done', flush=True)
