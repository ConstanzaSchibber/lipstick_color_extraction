"""Data validation: catch broken/blank/unusable downloads and normalize
survivors to a clean, uniformly-encoded JPEG set (notebook 02's deterministic
steps). Notebook 02 owns the orchestration (looping, archiving
invalid_images.csv, printing per-step audit counts) and calls into these.
"""
import os

import numpy as np
import pandas as pd
from PIL import Image

FORMAT_TO_EXT = {
    'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp',
    'GIF': '.gif', 'BMP': '.bmp', 'TIFF': '.tiff',
}

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
BLANK_ALPHA_THRESHOLD = 20  # mean alpha below this → mostly transparent (0-255 scale)
BLANK_STD_THRESHOLD = 8     # pixel std dev below this → near-uniform/blank (0-255 scale)


def repair_web_extension(df, imgs_dir, img_name_col='img_name'):
    """Rename '.web'-suffixed files (a truncated/garbled download extension)
    to their real extension, detected via PIL's own format sniffing, and
    update the matching img_name so later steps see the real filename.

    Returns (df, stats) where stats has 'web_found', 'repaired', 'unreadable'
    counts -- mirrors src/dedup.py's split of computation vs. reporting.
    """
    repaired, unreadable = 0, 0
    web_idx = df.index[df[img_name_col].str.endswith('.web', na=False)].tolist()

    for i in web_idx:
        old_name = df[img_name_col][i]
        path = os.path.join(imgs_dir, old_name)
        try:
            im = Image.open(path)
            ext = FORMAT_TO_EXT.get(im.format)
            if ext is None:
                unreadable += 1
                continue
            new_name = os.path.splitext(old_name)[0] + ext
            os.rename(path, os.path.join(imgs_dir, new_name))
            df.at[i, img_name_col] = new_name
            repaired += 1
        except Exception:
            unreadable += 1

    stats = {'web_found': len(web_idx), 'repaired': repaired, 'unreadable': unreadable}
    return df, stats


def validate_image(img_name, imgs_dir):
    """Returns None if img_name is a valid, non-blank, supported-format image
    on disk in imgs_dir, or a string describing why it isn't."""
    path = os.path.join(imgs_dir, img_name)
    if not os.path.exists(path):
        return 'missing: file not on disk'
    ext = os.path.splitext(img_name)[1].lower()
    if ext not in VALID_EXTENSIONS:
        return f'unsupported format: {ext} extension'
    try:
        im = Image.open(path)
    except Exception as e:
        return type(e).__name__ + ': ' + str(e)
    if im.mode in ('RGBA', 'LA', 'PA'):
        alpha = np.array(im.getchannel('A'))
        if alpha.mean() < BLANK_ALPHA_THRESHOLD:
            return f'blank: mostly transparent (mean alpha={alpha.mean():.1f})'
    try:
        rgb = np.array(im.convert('RGB'))
        if rgb.std() < BLANK_STD_THRESHOLD:
            return f'blank: near-uniform color (std={rgb.std():.1f})'
    except Exception as e:
        return type(e).__name__ + ': ' + str(e)
    return None


def flag_unusable(df, imgs_dir, validation_col='validation_error',
                   min_file_size=1024, min_dim=10, img_name_col='img_name'):
    """Second-pass check (file size / pixel dimensions) over rows not already
    flagged by validate_image. Mutates and returns df[validation_col]."""
    for i in df.index:
        if pd.notna(df.at[i, validation_col]):
            continue  # already flagged
        path = os.path.join(imgs_dir, df[img_name_col][i])
        if os.path.getsize(path) < min_file_size:
            df.at[i, validation_col] = f'unusable: file size < {min_file_size} bytes'
            continue
        try:
            w, h = Image.open(path).size
            if w < min_dim or h < min_dim:
                df.at[i, validation_col] = f'unusable: dimensions too small ({w}x{h})'
        except Exception:
            pass  # already caught by validate_image
    return df[validation_col]


def open_on_white(path):
    """Load image composited on white background — handles palette PNGs with transparency.
    Direct .convert('RGB') on transparent PNGs fills transparent pixels with the palette
    color (often green), not white. This composites correctly before saving as JPEG."""
    img = Image.open(path)
    if img.mode in ('P', 'RGBA', 'LA') or 'transparency' in img.info:
        rgba = img.convert('RGBA')
        white = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        white.paste(rgba, mask=rgba.split()[3])
        return white.convert('RGB')
    return img.convert('RGB')
