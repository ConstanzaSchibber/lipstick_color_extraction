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

## `annotation_sample/` — 208 files

**Created by:** Notebook 2C (`2C_DataAnnotationImageRecognition.ipynb`)

208 images copied from `original_clean/` for annotation in Label Studio (14 of the 222 sampled were missing from `original_clean/` due to validation failures in 1B). Each image was labeled with one of five categories (`swatch`, `bullet`, `liquid`, `closed`, `color_not_shown`) and a brush mask over the color-showing area. Annotations exported to `data/processed/annotations_label_studio.json` and parsed into `data/processed/annotations.csv`.

---

## `annotation_sample_closed/` — 40 files

**Created by:** Notebook 2C (`2C_DataAnnotationImageRecognition.ipynb`)

Oversampling batch to increase coverage of the `closed` container class, which had only 6 examples in the initial annotation round. Images were selected by same-product-line matching: the 4 product lines already confirmed as `closed` were searched for unannotated shades (53 candidates total), and 40 were drawn proportionally across product lines. Intended for a second Label Studio annotation round using the same label schema as `annotation_sample/`.

---

## `groundtruth_old/` — 337 files

**Origin:** Previous multi-category project (blush, lipgloss, lipliner, lipstick combined)

Ground truth sample from before this project was narrowed to lipstick only. Superseded by `groundtruth/`. Kept for reference but not used by any current notebook.
