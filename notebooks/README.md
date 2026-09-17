# Notebooks

This folder contains the notebooks for the project, organized in execution order.

> **Just want to see the app?** Try it at [lipstickbycolor.github.io](https://lipstickbycolor.github.io/). Source code at [github.com/LipstickByColor](https://github.com/LipstickByColor).

---

## Setup

**1. Install dependencies:**
```
pip install -r requirements.txt
```

**2. Data files by stage:**

| Notebook | Reads | Writes |
|----------|-------|--------|
| 01_data_engineering | raw metadata CSVs | `products_with_images.csv` |
| 02_data_validation | `products_with_images.csv` | cleaned `products_with_images.csv` |
| 03_a_training_set_strategy | `products_with_images.csv` | `annotation_sample/s1_color_taxonomy.csv`, `s2_style_discovery.csv`, `s3_closed_containers.csv` |
| 03_b_training_annotation_color | annotated masks | CIELAB color labels for training |
| 03_c_training_annotation_image_recognition | training set CSVs | images for Label Studio |
| 04_a_validation_set_strategy | `products_with_images.csv` + training/active-learning CSVs (exclusion set) | `annotation_sample/eval_worklist.csv` |
| 04_b_validation_annotation_image | `eval_worklist.csv`, Label Studio export (brush masks) | `annotations/labels_val.csv` |
| 04_c_validation_evaluation | `labels_val.csv`, notebook 06's checkpoints | prints metrics, displays plots — writes no files |
| 05_test_set_strategy | placeholder — not yet implemented | — |
| 06_model_image_recognition | annotated labels + images | trained ResNet-18 classifier, U-Net segmenters |
| 07_active_learning (WIP) | `labels.csv`, checkpoints, catalog images | `active_learning_queue.csv`, `active_learning_seg_queue.csv`, `resnet18_classifier_al.pth` |
| 08_pipeline_inference | classifier + all images | `products_pipeline.csv` |
| 09_viz_cielab | `products_pipeline.csv` | visualizations |

Images should be placed in `data/img/original/`.

---

## Notebooks

**01. [Data Engineering](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/01_data_engineering.ipynb)**
- Collect product metadata from retailers via API and web scraping
- Validate URLs and download product images
- Exploratory data analysis: product categories, image size, resolution, color distributions

> **To run this notebook**, use the sample metadata file included in the repo: `data/product_metadata/product_lipstick_metadata_sample.csv` (173 products). The full scraped dataset is not included.

**02. [Data Validation](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/02_data_validation.ipynb)**
- Check all downloaded images for corruption and readability
- Remove invalid files; output a clean image set for downstream processing

**03a. [Training Set Strategy](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/03_a_training_set_strategy.ipynb)**
- Build an annotation queue designed to be informative rather than representative: cover every region of the input space the model needs to handle, even rare ones
- Strategy 1 — Color taxonomy: consolidate 200+ raw `parent_color` values into 18 groups via a keyword-based LLM-assisted taxonomy; stratified sample with a floor of 5 per group (~223 images)
- Strategy 2 — Embedding-based style discovery: embed all unlabeled images with ResNet-50, cluster with BisectingKMeans, sample from visually under-covered clusters to capture rare photography styles not surfaced by metadata
- Strategy 3 — Rare-type oversampling: target product lines confirmed as `closed` containers (smallest class) via product-line matching to boost representation of that image type

**03b. [Training Annotation: Color](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/03_b_training_annotation_color.ipynb)**
- Extract CIELAB reference color labels from annotated masks for training images
- Visualize color coverage as a swatch grid grouped by color taxonomy
- Assess color space coverage: mean pairwise ΔE = 30.5 across the labeled set

**03c. [Training Annotation: Image Recognition](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/03_c_training_annotation_image_recognition.ipynb)**
- Prepare training images for Label Studio annotation
- Each image receives a presentation-type label (`swatch`, `bullet`, `liquid`, `closed`, `color_not_shown`) and, where applicable, a pixel-level segmentation mask over the color-bearing region
- Image type labels train the Stage 1 classifier; masks train the Stage 2 segmenters

**04a. [Validation Set Strategy](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/04_a_validation_set_strategy.ipynb)**
- Held-out pool = catalog minus every image seen by training, model selection, or active learning
- Type stratification via visual clustering (ResNet-50 embed → PCA → BisectingKMeans, same method as 03a), labeled by eye — not the classifier being evaluated
- Per-class sample sizes derived statistically (Cochran / mean-precision + finite-population correction); a representative *core* sample proportional to type, plus a separately-reported *type_boost* top-up for rare types
- Writes `eval_worklist.csv` for manual annotation in Label Studio

**04b. [Validation Annotation: Ingestion](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/04_b_validation_annotation_image.ipynb)**
- Parses the Label Studio export of `eval_worklist.csv` (type labels + brush masks)
- Ground-truth CIELAB is the median color inside the annotator's own brush mask, applied to the original image — no separate manually-cropped swatch, unlike notebook 03_b's training-set ground truth
- Filters to the validation-round annotations by date (the same Label Studio project also holds older training-set annotations)
- Resolves Label Studio's filename-truncation bug by hashing pixel content against its local media store, and drops redundant rows for pixel-identical duplicate images
- Writes `labels_val.csv` — the ground-truth table notebook 04c scores against

**04c. [Validation Evaluation](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/04_c_validation_evaluation.ipynb)**
- Scores notebook 06's classifier and segmenters against `labels_val.csv`
- Reports per-class classifier accuracy/FN/FP rate (Wilson interval), segmentation IoU, and ΔE CIE 2000 vs. ground truth (mean/median with a $t$ interval, plus the ≤2.3/2.3–5/>5 JND tier breakdown)

> **One break from strict numeric order.** 04c needs notebook 06's trained
> checkpoints, which don't exist until *after* this point in the pipeline —
> run 04c once notebook 06 and 04b are both done.

**05. [Test Set Strategy](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/05_test_set_strategy.ipynb)**
- Placeholder — not yet implemented

**06. [Model: Image Recognition & Segmentation](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/06_model_image_recognition.ipynb)**
- Fine-tune ResNet-18 to classify product images into five types: `swatch`, `bullet_lipstick`, `liquid_lipstick`, `closed`, `color_not_shown`
- Train two U-Nets (ResNet-18 encoder) for color-region segmentation: one for bullet/liquid, one for closed containers
- Apply type-conditional color extraction: each product type routes to the appropriate extraction method
- Active learning cycle: surface low-confidence classifier predictions, correct labels, retrain Stage 1

**07. [Active Learning](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/07_active_learning.ipynb)** *(work in progress — not yet runnable standalone)*
- Part 1: score the Stage 1 classifier's uncertainty on out-of-training images, export a review queue, apply human corrections, retrain
- Part 2: score segmentation-mask uncertainty on images the classifier is already confident about, prioritize those for mask annotation
- Split out from notebook 06's active-learning sections into its own notebook; still assumes some objects (`clf`, `CLASSES`, checkpoints) are already in memory from notebook 06

> **Archived:** the k-means vs. segmentation strategy-comparison notebook moved to
> [`old_notebooks/07_model_clustering.ipynb`](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/old_notebooks/07_model_clustering.ipynb).
> The k-means extraction method itself is still used in production — see notebook 08.

**08. [Pipeline: Inference](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/08_pipeline_inference.ipynb)**
- Apply the full two-stage pipeline to all ~9k product images
- Stage 1: classify each image with the ResNet-18 classifier
- Stage 2: extract color using the type-appropriate method (clustering or swatch extraction)
- Output: `products_pipeline.csv` with L, a, b, and hex values for every product

**09. [Visualization: CIELAB](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/09_viz_cielab.ipynb)**
- Visualize the distribution of extracted colors across the CIELAB color space
- a\*–b\* scatter plot (chromatic plane) and L\* distribution (lightness)
- Faceted views by product format (swatch, bullet, liquid)
