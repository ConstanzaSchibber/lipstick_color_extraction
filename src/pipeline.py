"""Type-conditional routing: which segmenter and which color-extraction
method a presentation type uses (see the README's "Evaluation & Production
Routing" section).

- `bullet`, `liquid`, `pencil` share one segmenter, extracted by plain median.
- `closed` (transparent containers) uses the dominant-cluster extraction,
  since the visible color region can include glare/reflection pixels a plain
  median would be skewed by.
- `swatch` and `lips` each get their own segmenter, extracted by plain median.
- Anything else (`unclassifiable`, or an unrecognized label) has no color to
  extract.
"""
import numpy as np
from PIL import Image

from src.segment import predict_mask
from src.extract_color import extract_masked_color, extract_dominant_cluster_color

SEG_CLASSES = {"bullet", "liquid", "pencil"}


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
