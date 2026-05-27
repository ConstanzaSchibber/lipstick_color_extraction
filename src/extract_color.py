"""Routing + color extraction for Model C.

Given a product image, classifies its type and extracts the lipstick color as
median LAB values using the appropriate strategy per type.
"""
import numpy as np
from PIL import Image
from skimage import color as skcolor

from src.classify import load_classifier, predict as clf_predict
from src.segment import load_segmenter, predict_mask


def extract_swatch_color(img_rgb: np.ndarray) -> np.ndarray:
    """Median LAB after removing white (L>95) and black (L<5) background pixels."""
    img_lab = skcolor.rgb2lab(img_rgb / 255.0)
    L = img_lab[:, :, 0]
    fg = (L > 5) & (L < 95)
    if fg.sum() == 0:
        fg = np.ones_like(L, dtype=bool)
    return np.median(img_lab[fg], axis=0)


def extract_masked_color(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Median LAB of pixels inside the binary mask."""
    img_lab = skcolor.rgb2lab(img_rgb / 255.0)
    pixels = img_lab[mask == 1]
    if len(pixels) == 0:
        return np.full(3, np.nan)
    return np.median(pixels, axis=0)


class ColorExtractor:
    """End-to-end two-stage color extractor.

    Args:
        classifier_path: Path to resnet18_classifier.pth checkpoint.
        segmenter_path:  Path to unet_segmenter.pth checkpoint.
        device:          'cpu', 'cuda', or 'mps'.
    """

    SEG_CLASSES = {"bullet_lipstick", "liquid_lipstick"}

    def __init__(self, classifier_path: str, segmenter_path: str, device: str = "cpu"):
        self.clf = load_classifier(classifier_path, device)
        self.seg = load_segmenter(segmenter_path, device)
        self.device = device

    def extract(self, img_path: str) -> dict:
        """Return a dict with product_type, confidence, L, a, b.

        L/a/b are None for 'other' type (caller should fall back to Model A).
        """
        product_type, confidence = clf_predict(self.clf, img_path, self.device)
        img = np.array(Image.open(img_path).convert("RGB"))

        if product_type == "swatch":
            lab = extract_swatch_color(img)
        elif product_type in self.SEG_CLASSES:
            mask = predict_mask(self.seg, img_path, self.device)
            lab = extract_masked_color(img, mask)
        else:
            lab = np.full(3, np.nan)

        return {
            "product_type": product_type,
            "confidence": confidence,
            "L": float(lab[0]) if not np.isnan(lab[0]) else None,
            "a": float(lab[1]) if not np.isnan(lab[1]) else None,
            "b": float(lab[2]) if not np.isnan(lab[2]) else None,
        }
