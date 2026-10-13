# -*- coding: utf-8 -*-
"""
Multi-layer network construction and local community detection (LCD)
refinement for vessel probability maps.

Three graph layers over SLIC superpixels:
  L1 appearance  (colour/intensity similarity)
  L2 spatial     (centroid proximity)
  L3 semantic    (vessel-probability similarity)
plus the skeleton-endpoint community bridging post-processor (lcd_reconnect).
"""
import numpy as np
from scipy import sparse
from skimage.segmentation import slic
from skimage.morphology import skeletonize
from skimage.graph import route_through_array


# --------------------------------------------------------------------------- #
#  Multi-layer superpixel graph                                               #
# --------------------------------------------------------------------------- #
def build_multilayer_graph(im, prob, vess, fov, n_sp=800):
    """Return (segments, [A_app, A_spa, A_sem]) sparse layer adjacencies."""
    fovm = fov > 0
    seg = slic(im, n_segments=n_sp, compactness=10, mask=fovm,
               start_label=0, channel_axis=-1)
    n = seg.max() + 1
    flat = seg.ravel()
    # superpixel feature aggregates
    cnt = np.bincount(flat, minlength=n).clip(min=1)
    feats = []
    for ch in range(im.shape[2]):
        feats.append(np.bincount(flat, weights=im[..., ch].ravel(), minlength=n) / cnt)
    pmean = np.bincount(flat, weights=prob.ravel(), minlength=n) / cnt
    vmean = np.bincount(flat, weights=vess.ravel(), minlength=n) / cnt
    ys, xs = np.indices(seg.shape)
    cy = np.bincount(flat, weights=ys.ravel(), minlength=n) / cnt
    cx = np.bincount(flat, weights=xs.ravel(), minlength=n) / cnt

    # adjacency between touching superpixels
    edges = set()
    for a, b in ((seg[:, :-1], seg[:, 1:]), (seg[:-1, :], seg[1:, :])):
        m = a != b
        for u, v in zip(a[m].ravel(), b[m].ravel()):
            edges.add((min(u, v), max(u, v)))
    U = np.array([e[0] for e in edges]); V = np.array([e[1] for e in edges])

    F = np.stack(feats, 1)                                # (n, 3) colour
    d_col = np.exp(-(((F[U] - F[V]) ** 2).sum(1)) / 0.02)
    d_spa = np.exp(-(((cy[U] - cy[V]) ** 2 + (cx[U] - cx[V]) ** 2)) / (2 * 15 ** 2))
    d_sem = np.exp(-((pmean[U] - pmean[V]) ** 2 + (vmean[U] - vmean[V]) ** 2) / 0.05)

    A_app = sparse.csr_matrix((d_col, (U, V)), shape=(n, n))
    A_spa = sparse.csr_matrix((d_spa, (U, V)), shape=(n, n))
    A_sem = sparse.csr_matrix((d_sem, (U, V)), shape=(n, n))
    layers = [A + A.T for A in (A_app, A_spa, A_sem)]
    return seg, layers, pmean


# --------------------------------------------------------------------------- #
#  LCD v1: modularity-gain community expansion                                #
# --------------------------------------------------------------------------- #
def lcd_refine(prob, fov, seg, layers, pmean, thr=0.5, alpha=(1., 1., 1.)):
    """Grow vessel communities from high-confidence seed superpixels by
    greedy multi-layer modularity gain; refine pixel probabilities."""
    A = alpha[0] * layers[0] + alpha[1] * layers[1] + alpha[2] * layers[2]
    A = A.tocsr()
    seeds = np.where(pmean > thr + 0.15)[0]
    comm = np.zeros_like(pmean, dtype=int)  # community id per node, 0=none
    cid = 0
    indptr, indices, data = A.indptr, A.indices, A.data
    for s in seeds:
        if comm[s] != 0:
            continue
        cid += 1
        members = {s}
        comm[s] = cid
        frontier = [s]
        while frontier:
            u = frontier.pop()
            for k in range(indptr[u], indptr[u + 1]):
                v, w = indices[k], data[k]
                if comm[v] == 0 and w > 0.25 and pmean[v] > thr - 0.2:
                    comm[v] = cid
                    members.add(v)
                    frontier.append(v)
    refined = prob.copy()
    for c in range(1, cid + 1):
        nodes = np.where(comm == c)[0]
        if len(nodes) < 2:
            continue
        mu = pmean[nodes].mean()
        if mu > thr:
            for nd in nodes:
                m = seg == nd
                refined[m] = np.maximum(refined[m], min(mu, 0.9))
    return refined, comm


# --------------------------------------------------------------------------- #
#  LCD v2: label diffusion + personalised PageRank                            #
# --------------------------------------------------------------------------- #
def lcd_refine2(prob, fov, seg, layers, pmean, thr=0.5, restart=0.15):
    """Personalised-PageRank diffusion of vessel labels over the fused graph."""
    A = (layers[0] + layers[1] + 2 * layers[2]).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    P = A.multiply(1 / deg[:, None]).tocsr()
    pers = (pmean > thr + 0.15).astype(float)
    pers /= max(pers.sum(), 1)
    r = pers.copy()
    for _ in range(50):
        r_new = (1 - restart) * P.T @ r + restart * pers
        if np.abs(r_new - r).sum() < 1e-6:
            r = r_new
            break
        r = r_new
    refined = prob.copy()
    boost = np.clip(r / max(r.max(), 1e-9), 0, 1) * 0.3
    for nd in range(len(pmean)):
        if boost[nd] > 0.05 and pmean[nd] > thr - 0.25:
            m = seg == nd
            refined[m] = np.maximum(refined[m], prob[m] + boost[nd])
    return refined, r


# --------------------------------------------------------------------------- #
#  LCD v3: skeleton-endpoint community bridging                               #
# --------------------------------------------------------------------------- #
def skeleton_endpoints(sk):
    """Return endpoint coordinates of a binary skeleton."""
    sk = sk.astype(np.uint8)
    k = np.ones((3, 3), int)
    from scipy.ndimage import convolve
    nb = convolve(sk, k, mode='constant') - sk
    ep = np.argwhere(sk & (nb == 1))
    return ep


def lcd_reconnect(prob, fov, thr=0.5, radius=35, path_thr=2.2,
                  min_comp=0, boost_val=0.52):
    """Bridge broken vessel communities: for every pair of skeleton endpoints
    closer than `radius`, route a minimum-cost path through the inverse
    probability landscape; accept the bridge if its mean cost < path_thr.

    Returns (refined_prob, refined_mask, added_paths).
    """
    from scipy import ndimage as ndi
    fovm = fov > 0
    M = (prob > thr) & fovm
    if min_comp > 0:
        lab, n = ndi.label(M)
        sizes = np.bincount(lab.ravel())
        kill = np.where(sizes < min_comp)[0]
        kill = kill[kill > 0]
        if len(kill):
            M[np.isin(lab, kill)] = False
    sk = skeletonize(M)
    eps = skeleton_endpoints(sk)
    P2 = prob.copy()
    added = []
    used = set()
    cost = np.exp(-6 * prob) + 0.01
    cost[~fovm] = 1e6
    for i, (y, x) in enumerate(eps):
        if i in used:
            continue
        d = np.hypot(eps[:, 0] - y, eps[:, 1] - x)
        cand = np.where((d > 0) & (d < radius))[0]
        if len(cand) == 0:
            continue
        j = cand[np.argmin(d[cand])]
        if j in used:
            continue
        y2, x2 = eps[j]
        try:
            path, pcost = route_through_array(cost, (y, x), (y2, x2),
                                              fully_connected=True)
        except Exception:
            continue
        path = np.array(path)
        if len(path) < 2:
            continue
        mcost = pcost / len(path)
        if mcost < path_thr:
            for py, px in path:
                if fovm[py, px]:
                    P2[py, px] = max(P2[py, px], boost_val)
            added.append(path)
            used.add(i)
            used.add(j)
    M2 = P2 > thr
    return P2, M2, added
