"""Color extraction from a segmentation mask.

Given an RGB image and a binary color-region mask (from src/segment.py),
extracts a single representative CIELAB color. Two strategies, chosen by
presentation type at the call site (see 07_end_to_end_evaluation.ipynb's
route_and_extract and 08_pipeline_inference.ipynb):

- extract_masked_color: plain median LAB — used for most types.
- extract_dominant_cluster_color: median of the largest k-means cluster —
  used for `closed` (transparent-container) images, where the visible color
  region can include glare/reflection pixels a plain median would be skewed
  by.
"""
import numpy as np
from skimage.color import rgb2lab
from sklearn.cluster import KMeans


def extract_masked_color(img_rgb: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
    """Median LAB of the pixels inside the binary mask."""
    lab = rgb2lab(img_rgb / 255.0)
    px = lab[mask_arr == 1]
    return np.full(3, np.nan) if len(px) == 0 else np.median(px, axis=0)


def extract_dominant_cluster_color(
    img_rgb: np.ndarray,
    mask_arr: np.ndarray,
    k: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """Median LAB of the largest of k color clusters inside the mask."""
    lab = rgb2lab(img_rgb / 255.0)
    px = lab[mask_arr == 1]
    if len(px) == 0:
        return np.full(3, np.nan)
    if len(px) < k:
        return np.median(px, axis=0)
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit(px)
    return km.cluster_centers_[np.bincount(km.labels_).argmax()]
