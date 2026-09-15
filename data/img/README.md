# Image Folders

## `original/` — 9,488 files (mixed formats)

**Created by:** notebook 01 (`01_data_engineering.ipynb`)
**Validated by:** notebook 02 (`02_data_validation.ipynb`)

All product images downloaded from retailer URLs. Mixed formats: `.jpg`, `.png`, `.webp`, `.jpe`, `.tif`, `.gif`.

Notebook 02 flags invalid images (missing, blank, transparent, too small) in `data/processed/products_with_images.csv` — those files remain on disk but are excluded from all downstream notebooks via the CSV.

---

## `original_clean/` — 9,153 files (all `.jpg`)

**Created by:** notebook 02 (`02_data_validation.ipynb`)

All valid images from `data/processed/products_with_images.csv` converted to `.jpg`. Non-JPG formats from `original/` (`.webp`, `.jpe`, `.tif`, `.gif`) do not render reliably in browser-based tools like Label Studio.

This is the **source folder for all downstream notebooks and annotation work**. It is fully reproducible — delete it and re-run notebook 02 to regenerate.

---

## `groundtruth/` — 222 files

**Sampled by:** notebook 03_a (`03_a_training_set_strategy.ipynb`) — stratified sample copied here
**Annotated in:** notebook 03_b (`03_b_training_annotation_color.ipynb`) — manually cropped to isolate color swatches

222 images selected via stratified proportional sampling across 18 color groups. After copying, each image was manually cropped to show only the color swatch area. 209 of 222 were successfully cropped; 13 could not be annotated (transparent, blank, or ambiguous).

These files are **not identical to their counterparts in `original/`** — they have been cropped. The mean CIELAB of each cropped image is the ground truth label stored in `data/processed/products_with_images.csv` (`ground_truth_CIELAB`, `color_swatch == 1`).

---

## `annotation_sample/` — 338 files

**Created by:** notebook 03_c (`03_c_training_annotation_image_recognition.ipynb`)

The curated folder notebook 06 actually reads training images from (see `IMG_DIRS`). Meant to hold every image referenced in `annotations_combined.csv` — not the full 9k-image `original_clean/` universe. Started at 208 images (one presentation-type label + brush mask each, five original categories: `swatch`, `bullet`, `liquid`, `closed`, `color_not_shown`) and grew across several annotation rounds as `pencil`, `lips`, and `unclassifiable` were added and rare types were oversampled; notebook 03_c backfills anything referenced in the final combined annotations that isn't already copied here.

---

## `annotation_sample_closed/` — 53 files

**Created by:** notebook 03_a, Strategy 3 (`s3_closed_containers.csv`)

Oversampling batch targeting the `closed` container class (product color visible through transparent/window packaging), which had very few examples in the initial annotation round. Candidates were selected by same-product-line matching: product lines already confirmed as `closed` were searched for unannotated shades, then sampled proportionally across lines.

---

## `annotation_sample_style/` — 54 files

**Created by:** notebook 03_a, Strategy 2 (`s2_style_discovery.csv`)

Embedding-based style-discovery oversampling: all unlabeled images embedded with ResNet-50, clustered, and images sampled from visually under-covered clusters to capture rare photography styles/packaging formats not surfaced by metadata-based strategies.

---

## `annotation_sample_pencil/` — 40 files

**Created by:** notebook 03_a, Strategy 4 (`s4_pencil_crayon.csv`)

Keyword-oversampling batch targeting `pencil`/lip-crayon products: searched `products_with_images.csv` for "pencil"/"crayon" in the product name and sampled unannotated candidates across product lines.

---

## `annotation_sample_lips_notshown/` — 37 files

**Created by:** notebook 03_c, Round 4 oversampling

Batch targeting the `lips` (product shown on lips) and `color_not_shown`/`unclassifiable` types. Staged for review but not fully copied into `annotation_sample/` — most of these remain pending manual annotation (tracked in `data/annotations/lips_notshown_pending.txt`).

---

## `annotation_sample_color_strat/` — 208 files

**Legacy.** Staged copy from an earlier version of notebook 03_a's Strategy 1 (color taxonomy), before that strategy was simplified to sample directly from `original_clean/` rather than maintaining its own copied folder. Superseded — no notebook samples from here anymore, but it's kept as a Label Studio filename-resolution staging dir (notebook 03_c hashes files here to recover truncated upload names).

---

## `groundtruth_old/` — 337 files

**Origin:** previous multi-category project (blush, lipgloss, lipliner, lipstick combined)

Ground truth sample from before this project was narrowed to lipstick only. Superseded by `groundtruth/`. Kept for reference but not used by any current notebook.

---

## Visualizations

`product_type_distribution.png` and `annotation_label_distribution.png` / `annotation_label_distribution_combined.png` are plots saved by notebook 03_c (presentation-type and annotation-label distributions), not image data.
