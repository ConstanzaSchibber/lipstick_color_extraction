# Images

The images themselves are not included in this repository (they're gitignored). This
folder is where the pipeline stores them locally.

- **Product images** — every lipstick product image downloaded from retailer listings
  (notebook 01), plus a cleaned copy with invalid images removed and everything
  converted to `.jpg` (notebook 02). The cleaned copy is what all later notebooks read.
- **Annotation images** — the subsets sampled for manual labeling in Label Studio:
  the training set (notebooks 03_a–03_c) and the validation set (notebooks 04_a–04_b).
