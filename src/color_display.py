"""Rendering an extracted CIELAB color for display: a hex string and the
small HTML color-circle swatch used by 08_pipeline_inference.ipynb's output
columns (hex_color, Circle, group_hex, group_circle), which the app consumes
directly from products_pipeline.csv.
"""
import numpy as np
from skimage.color import lab2rgb


def lab_to_hex(L: float, a: float, b: float) -> str:
    """Convert a single CIELAB triple to a '#rrggbb' hex string."""
    rgb = lab2rgb(np.array([[[L, a, b]]])).clip(0, 1)[0, 0]
    r, g, bv = (rgb * 255).round().astype(int)
    return f'#{r:02x}{g:02x}{bv:02x}'


def make_circle(hex_color: str, size: int = 20) -> str:
    """A small HTML div rendered as a solid-color circle, for notebook/table display."""
    return (f'<div style="width:{size}px; height:{size}px; '
            f'border-radius:50%; background-color:{hex_color};"></div>')
