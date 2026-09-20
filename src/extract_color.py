"""Color extraction from a segmentation mask.

Given an RGB image and a binary color-region mask (from src/segment.py),
extracts a single representative CIELAB color. Strategies, chosen by
presentation type at the call site (see 07_end_to_end_evaluation.ipynb's
route_and_extract and 08_pipeline_inference.ipynb):

- extract_masked_color: plain median LAB — used for most types.
- extract_dominant_cluster_color: median of the largest k-means cluster —
  used for `closed` (transparent-container) images, where the visible color
  region can include glare/reflection pixels a plain median would be skewed
  by.
- extract_masked_color_dropping_near_black_component: `bullet` and `liquid`
  only — see its own docstring.
"""
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.measure import label
from sklearn.cluster import KMeans


def extract_masked_color(img_rgb: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
    """Median LAB of the pixels inside the binary mask."""
    lab = rgb2lab(img_rgb / 255.0)
    px = lab[mask_arr == 1]
    return np.full(3, np.nan) if len(px) == 0 else np.median(px, axis=0)


def extract_masked_color_dropping_near_black_component(
    img_rgb: np.ndarray,
    mask_arr: np.ndarray,
    l_thresh: float = 20.0,
    a_thresh: float = 5.0,
    delta_e_thresh: float = 25.0,
) -> np.ndarray:
    """Median LAB of the mask, but for `bullet`/`liquid`'s tube and
    bottle+wand product shots — where the shared bullet/liquid/pencil
    segmenter often predicts a second, disconnected mask blob on the black
    tube body or wand cap/handle in addition to the true color region —
    drops the near-black component(s) before taking the median, when the
    mask splits into components that are both:

    1. Dark + achromatic vs. not: a component is "near-black" only if its
       median color is both dark (L < l_thresh) *and* has a\\* within
       a_thresh of 0. a\\* (not overall chroma) is the discriminator: a real
       cap/handle is genuinely neutral plastic, and its median a\\* sat at
       -0.7 to 1.8 in every confirmed case, regardless of its b\\* (a mild
       blue cast from lighting is common and doesn't make it a real color).
       A first version of this rule used overall chroma < 15, but that
       misfired on a real dark, desaturated product color (a 3-wand-tip
       comparison photo's actual "Rich Forest Green" tip: L=12.5, a=-8.7,
       chroma=8.7 — chroma alone can't tell a low-saturation *hue* like dark
       green from true neutral black, but its a\\* of -8.7 is a real,
       nonzero hue direction, well outside every confirmed cap/handle's
       -0.7 to 1.8 range). Every confirmed genuinely-dark-but-real shade
       checked (wine, chocolate, plum, terracotta) had |a\\*| >= 14.
    2. Meaningfully different colors: the near-black component(s) and the
       rest must differ by more than delta_e_thresh (CIE2000) — two
       same-shade blobs split by lighting/glare (e.g. a glossy highlight)
       can drift apart in raw L/a without being a different color, and ΔE
       is this project's actual perceptual-difference metric (see
       CLAUDE.md). Observed gap: genuine same-shade splits were ΔE 7.5-19.2;
       real cap/handle artifacts were ΔE 35.4-51.1.

    Falls back to the plain mask median (all components) otherwise —
    single component, no near-black component, all components near-black,
    or a near-black component too similar in color to drop.
    """
    lab = rgb2lab(img_rgb / 255.0)
    labeled, n = label(mask_arr.astype(bool), connectivity=2, return_num=True)
    if n >= 2:
        near_black = {}
        for i in range(1, n + 1):
            med = np.median(lab[labeled == i], axis=0)
            near_black[i] = med[0] < l_thresh and abs(med[1]) < a_thresh
        dark_labels = [i for i, is_black in near_black.items() if is_black]
        light_labels = [i for i, is_black in near_black.items() if not is_black]
        if dark_labels and light_labels:
            dark_med = np.median(lab[np.isin(labeled, dark_labels)], axis=0)
            light_med = np.median(lab[np.isin(labeled, light_labels)], axis=0)
            if deltaE_ciede2000(dark_med, light_med) > delta_e_thresh:
                return light_med
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
