# Image Folders

## `original/` — 9,502 files (mixed formats)

**Created by:** Notebook 1A (`1A_DataEngineering.ipynb`)
**Validated by:** Notebook 1B (`1B_DataValidation.ipynb`)

All product images downloaded from retailer URLs. Mixed formats: `.jpg`, `.png`, `.webp`, `.jpe`, `.tif`, `.gif`. The `.web` files that existed after download were renamed to their correct extension by 1B.

1B flags invalid images (missing, blank, transparent, too small) in `data/processed/products_with_images.csv` — those files remain on disk but are excluded from all downstream notebooks via the CSV.

---

## `groundtruth/` — 222 files

**Created by:** Notebook 2A (`2A_DataAnnotatioSampling.ipynb`) — stratified sample copied here
**Annotated in:** Notebook 2B (`2B_DataAnnotationGT.ipynb`) — manually cropped to isolate color swatches

222 images selected via stratified proportional sampling across 18 color groups. After copying, each image was manually cropped to show only the color swatch area. 209 of 222 were successfully cropped; 13 could not be annotated (transparent, blank, or ambiguous).

These files are **not identical to their counterparts in `original/`** — they have been cropped. The mean CIELAB of each cropped image is the ground truth label stored in `data/processed/products_with_images.csv` (`ground_truth_CIELAB`, `color_swatch == 1`).

---

## `original_clean/` — 9,167 files (all `.jpg`)

**Created by:** Notebook 1B (`1B_DataValidation.ipynb`)

All valid images from `data/processed/products_with_images.csv` converted to `.jpg`. Non-JPG formats from `original/` (`.webp`, `.jpe`, `.tif`, `.gif`) do not render reliably in browser-based tools like Label Studio.

This is the **source folder for all downstream notebooks and annotation work**. It is fully reproducible — delete it and re-run the last cell of 1B to regenerate.

---

## `groundtruth_old/` — 337 files

**Origin:** Previous multi-category project (blush, lipgloss, lipliner, lipstick combined)

Ground truth sample from before this project was narrowed to lipstick only. Superseded by `groundtruth/`. Kept for reference but not used by any current notebook.
