# Data

This folder is where the pipeline stores its data locally. Most of it is not included
in this repository (it's gitignored). Only these files are committed:

- `product_metadata/product_lipstick_metadata_sample.csv`: a 173-product sample of
  the scraped metadata, with only the columns the pipeline uses (brand, product,
  shade, shade description, parent color and image URL). It's enough to run the
  notebooks end to end.
- `product_metadata/brand_tiers.json`: a rough price-tier classification of brands
  (drugstore, mid-range, luxury).

The rest of the folder holds:

- **`product_metadata/`**: the full scraped product metadata, which is the raw input
  to the pipeline. It isn't committed, to protect the scraped dataset.
- **`img/`**: product images and the images used for annotation (see
  [img/README.md](img/README.md)).
- **`annotations/`** and **`processed/`**: annotation labels, Label Studio exports,
  and the CSVs each notebook writes, ending with the per-product color output from
  notebook 08.

If you need the full dataset for research purposes, please open an issue.
