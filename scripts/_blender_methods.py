"""Shared blender techniques ported from f1-pit-stops-blender-0-95454.ipynb.

All functions operate on probability arrays and return clipped probability arrays.
The notebook uses these to blend two submission CSVs; here we use them to combine
our OOF (anchor) with another OOF (support) and score by OOF AUC.
"""
import numpy as np

CLIP_LOW = 1e-7
CLIP_HIGH = 1 - 1e-7


def clip_pred(p):
    return np.clip(np.asarray(p, dtype=float), CLIP_LOW, CLIP_HIGH)


def normalized_rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values))
    return ranks


def rank_blend(anchor, support, support_weight):
    a = normalized_rank(anchor)
    s = normalized_rank(support)
    blended = (1 - support_weight) * a + support_weight * s
    order = np.argsort(blended, kind="mergesort")
    out = np.empty_like(anchor, dtype=float)
    out[order] = np.sort(anchor)
    return clip_pred(out)


def logit_transform(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid_transform(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit_rank_blend(anchor, support, support_weight):
    a = logit_transform(normalized_rank(anchor))
    s = logit_transform(normalized_rank(support))
    blended_rank = sigmoid_transform((1 - support_weight) * a + support_weight * s)
    order = np.argsort(blended_rank, kind="mergesort")
    out = np.empty_like(anchor, dtype=float)
    out[order] = np.sort(anchor)
    return clip_pred(out)


def multi_tiered_gate(anchor, support, core_w=0.04, edge_w=0.01,
                      core_lo=0.15, core_hi=0.85, edge_lo=0.02, edge_hi=0.98):
    core = logit_rank_blend(anchor, support, core_w)
    edge = logit_rank_blend(anchor, support, edge_w)
    out = anchor.copy()
    core_mask = (anchor >= core_lo) & (anchor <= core_hi)
    edge_mask = ((anchor >= edge_lo) & (anchor < core_lo)) | ((anchor > core_hi) & (anchor <= edge_hi))
    out[core_mask] = core[core_mask]
    out[edge_mask] = edge[edge_mask]
    return clip_pred(out)


def rank_max_blend(anchor, support, support_weight):
    r_a = normalized_rank(anchor)
    r_s = normalized_rank(support)
    blended = np.maximum(r_a, r_s * support_weight)
    order = np.argsort(blended, kind="mergesort")
    out = np.empty_like(anchor, dtype=float)
    out[order] = np.sort(anchor)
    return clip_pred(out)


def rank_min_blend(anchor, support, support_weight):
    r_a = normalized_rank(anchor)
    r_s = normalized_rank(support)
    blended = np.minimum(r_a, r_s / support_weight)
    order = np.argsort(blended, kind="mergesort")
    out = np.empty_like(anchor, dtype=float)
    out[order] = np.sort(anchor)
    return clip_pred(out)


def piecewise_rescale(anchor, support, bins=20, scalar_clip=None):
    order = np.argsort(anchor, kind="mergesort")
    n = len(anchor)
    bin_size = n // bins
    out = anchor.copy()
    for i in range(bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < bins - 1 else n
        idx = order[start:end]
        a_mean = np.mean(anchor[idx])
        s_mean = np.mean(support[idx])
        scalar = (s_mean + 1e-9) / (a_mean + 1e-9)
        if scalar_clip is not None:
            scalar = np.clip(scalar, *scalar_clip)
        out[idx] = np.clip(out[idx] * scalar, CLIP_LOW, CLIP_HIGH)
    return clip_pred(out)
