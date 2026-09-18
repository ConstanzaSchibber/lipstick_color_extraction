"""Evaluation-metric helpers shared across notebooks: confidence intervals
(for a proportion and for a mean), color difference, and mask IoU.

Note: this mask_iou (two already-computed binary masks) is a different thing
from the batch_iou defined in 06_b_model_segmenters.ipynb (a batch of raw
model logits, thresholded and averaged) — same-sounding name, different job,
used at a different stage (training-time tracking vs. post-hoc evaluation).
Not consolidated on purpose; don't assume they're interchangeable.
"""
import numpy as np
from scipy import stats

# colormath needs a numpy attribute removed in newer numpy — restore it
# before import.
if not hasattr(np, "asscalar"):
    np.asscalar = lambda a: a.item()
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion k/n — more reliable
    than the normal approximation when n is small."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return phat, centre - half, centre + half


def mean_ci(x) -> tuple[float, float, float]:
    """Mean and 95% Student-t confidence interval for a 1-D array, ignoring
    NaNs. Returns (nan, nan, nan) if fewer than 2 valid values remain."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan, np.nan)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def delta_e(pred_lab, true_lab) -> float:
    """CIE2000 color difference between two Lab triples."""
    return float(delta_e_cie2000(LabColor(*pred_lab), LabColor(*true_lab)))


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-union between two binary masks."""
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union else np.nan
