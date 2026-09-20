"""Type-conditional routing: which segmenter and which color-extraction
method a presentation type uses (see the README's "Evaluation & Production
Routing" section).

- `bullet` and `liquid` share the "main" segmenter with `pencil`, and both
  drop any near-black, desaturated mask component before extracting — see
  extract_color.py's extract_masked_color_dropping_near_black_component
  docstring. Both are bottle/tube shots where the segmenter can predict a
  second mask blob on the black tube/handle body in addition to the true
  color region — confirmed on both (e.g. a Christian Louboutin bullet's
  glossy black cone, a Hera bullet's black tube) with the same signature:
  the extra blob is dark (L<20) *and* near-zero a* (a real neutral, not a
  dark-but-saturated shade), unlike every confirmed genuine dark shade
  checked in either class. `pencil` is NOT included — its own two-blob
  splits (e.g. a pencil tip + its swatch stroke) are both legitimately part
  of the product, not a packaging artifact to discard.
- `closed` (transparent containers) uses the dominant-cluster extraction,
  since the visible color region can include glare/reflection pixels a plain
  median would be skewed by.
- `swatch`, `lips`, and `pencil` each use plain median extraction (`pencil`
  shares the `main` segmenter; `swatch`/`lips` have their own).
- Anything else (`unclassifiable`, or an unrecognized label) has no color to
  extract.
"""
import numpy as np
from PIL import Image

from src.segment import predict_mask
from src.extract_color import (
    extract_masked_color,
    extract_dominant_cluster_color,
    extract_masked_color_dropping_near_black_component,
)

NEAR_BLACK_DROP_CLASSES = {"bullet", "liquid"}
SEG_CLASSES = {"pencil"}


def route_and_extract(img_path: str, route_label: str, segmenters: dict, device: str = "cpu"):
    """Route an image to its type's segmenter + extraction method.

    segmenters: dict of loaded U-Net models (see src.segment.load_segmenter),
    keyed by role — 'main' (bullet/liquid/pencil), 'closed', 'swatch', 'lips'.

    Returns (mask, lab): mask is the predicted binary mask (or None if the
    type has no segmenter), lab is the extracted CIELAB triple (or all-NaN).
    """
    img = np.array(Image.open(img_path).convert("RGB"))
    if route_label == "swatch":
        mask = predict_mask(segmenters["swatch"], img_path, device)
        lab = extract_masked_color(img, mask)
    elif route_label in NEAR_BLACK_DROP_CLASSES:
        mask = predict_mask(segmenters["main"], img_path, device)
        lab = extract_masked_color_dropping_near_black_component(img, mask)
    elif route_label in SEG_CLASSES:
        mask = predict_mask(segmenters["main"], img_path, device)
        lab = extract_masked_color(img, mask)
    elif route_label == "closed":
        mask = predict_mask(segmenters["closed"], img_path, device)
        lab = extract_dominant_cluster_color(img, mask)
    elif route_label == "lips":
        mask = predict_mask(segmenters["lips"], img_path, device)
        lab = extract_masked_color(img, mask)
    else:
        mask, lab = None, np.full(3, np.nan)
    return mask, lab
