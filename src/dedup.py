"""Detects products that share one seller-uploaded photo across multiple shades
(e.g. a single stock photo reused for every color in a line).

Notebook 01 downloads and saves a separate file per product row even when
several rows point at the same photo, so these duplicates aren't caught by
02_data_validation's img_name-based dedup -- that only catches literal
on-disk filename collisions, a different case (see CLAUDE.md). A group's
photo can't be trusted to represent any one member's own shade color, so
08_pipeline_inference excludes every row in a flagged group from color
extraction before running the full corpus through the models.
"""
import hashlib
import os
from typing import Optional

import pandas as pd


def _sha256_of(path: str) -> Optional[str]:
    """Return the sha256 hex digest of a file's bytes, or None if unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


class _UnionFind:
    """Minimal disjoint-set, used to merge the img_url-match and
    content-hash-match groupings into one final grouping."""
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def find_shared_photo_groups(meta: pd.DataFrame, img_dir: str) -> pd.DataFrame:
    """Group rows of meta that show the literal same seller photo.

    Two exact-match passes are unioned into one grouping (no perceptual /
    near-duplicate matching): identical `img_url`, and identical file
    content (sha256 over the bytes in img_dir) for rows whose URLs differ
    but whose downloaded images are byte-identical.

    Returns a copy of meta with two added columns:
      - photo_group_id: int, shared by every row depicting the same photo
        (every row gets one, including singleton groups)
      - is_shared_stock_photo: bool, True for rows in a group spanning more
        than one distinct product `id` -- i.e. groups where the photo can't
        be trusted to represent any single member's own shade
    """
    df = meta.reset_index(drop=True).copy()
    uf = _UnionFind(len(df))

    for _, idx in df.groupby('img_url').groups.items():
        idx = list(idx)
        for i in idx[1:]:
            uf.union(idx[0], i)

    df['_content_hash'] = [
        _sha256_of(os.path.join(img_dir, name)) if isinstance(name, str) else None
        for name in df['img_name']
    ]
    for _, idx in df[df['_content_hash'].notna()].groupby('_content_hash').groups.items():
        idx = list(idx)
        for i in idx[1:]:
            uf.union(idx[0], i)
    df = df.drop(columns=['_content_hash'])

    roots = pd.array([uf.find(i) for i in range(len(df))])
    df['photo_group_id'] = pd.factorize(roots)[0]
    df['is_shared_stock_photo'] = df.groupby('photo_group_id')['id'].transform('nunique') > 1
    return df


def duplicate_report(df: pd.DataFrame) -> dict:
    """Summary counts for the shared-stock-photo groups found by find_shared_photo_groups."""
    flagged = df[df['is_shared_stock_photo']]
    return {
        'total_products': len(df),
        'flagged_products': len(flagged),
        'flagged_pct': 100 * len(flagged) / len(df) if len(df) else 0.0,
        'n_groups': flagged['photo_group_id'].nunique(),
    }
