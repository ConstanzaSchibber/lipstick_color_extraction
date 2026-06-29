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
| 03_training_set_strategy | `products_with_images.csv` | `annotation_sample/s1_color_taxonomy.csv`, `s2_style_discovery.csv`, `s3_closed_containers.csv` |
| 04_training_annotation_color | annotated masks | CIELAB color labels for training |
| 05_training_annotation_image_recognition | training set CSVs | images for Label Studio |
| 06_model_image_recognition | annotated labels + images | trained ResNet-18 classifier, U-Net segmenters |
| 07_model_clustering | classifier + images | evaluated clustering results |
| 08_pipeline_inference | classifier + all images | `products_pipeline.csv` |
| 09_viz_cielab | `products_pipeline.csv` | visualizations |
| 10_validation_test_set_strategy | `products_with_images.csv` + training set CSVs | validation and test set CSVs |

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

**03. [Training Set Strategy](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/03_training_set_strategy.ipynb)**
- Build an annotation queue designed to be informative rather than representative: cover every region of the input space the model needs to handle, even rare ones
- Strategy 1 — Color taxonomy: consolidate 200+ raw `parent_color` values into 18 groups via a keyword-based LLM-assisted taxonomy; stratified sample with a floor of 5 per group (~223 images)
- Strategy 2 — Embedding-based style discovery: embed all unlabeled images with ResNet-50, cluster with BisectingKMeans, sample from visually under-covered clusters to capture rare photography styles not surfaced by metadata
- Strategy 3 — Rare-type oversampling: target product lines confirmed as `closed` containers (smallest class) via product-line matching to boost representation of that image type

**04. [Training Annotation: Color](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/04_training_annotation_color.ipynb)**
- Extract CIELAB reference color labels from annotated masks for training images
- Visualize color coverage as a swatch grid grouped by color taxonomy
- Assess color space coverage: mean pairwise ΔE = 30.5 across the labeled set

**05. [Training Annotation: Image Recognition](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/05_training_annotation_image_recognition.ipynb)**
- Prepare training images for Label Studio annotation
- Each image receives a presentation-type label (`swatch`, `bullet`, `liquid`, `closed`, `color_not_shown`) and, where applicable, a pixel-level segmentation mask over the color-bearing region
- Image type labels train the Stage 1 classifier; masks train the Stage 2 segmenters

**06. [Model: Image Recognition & Segmentation](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/06_model_image_recognition.ipynb)**
- Fine-tune ResNet-18 to classify product images into five types: `swatch`, `bullet_lipstick`, `liquid_lipstick`, `closed`, `color_not_shown`
- Train two U-Nets (ResNet-18 encoder) for color-region segmentation: one for bullet/liquid, one for closed containers
- Apply type-conditional color extraction: each product type routes to the appropriate extraction method
- Active learning cycle: surface low-confidence classifier predictions, correct labels, retrain Stage 1

**07. [Model: Clustering](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/07_model_clustering.ipynb)**
- K-means clustering to extract dominant colors, routed by the Stage 1 classifier from notebook 06
- Filters out near-black and near-white clusters (background/packaging noise)
- Evaluate against ground truth labels using Delta E

**08. [Pipeline: Inference](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/08_pipeline_inference.ipynb)**
- Apply the full two-stage pipeline to all ~9k product images
- Stage 1: classify each image with the ResNet-18 classifier
- Stage 2: extract color using the type-appropriate method (clustering or swatch extraction)
- Output: `products_pipeline.csv` with L, a, b, and hex values for every product

**09. [Visualization: CIELAB](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/09_viz_cielab.ipynb)**
- Visualize the distribution of extracted colors across the CIELAB color space
- a\*–b\* scatter plot (chromatic plane) and L\* distribution (lightness)
- Faceted views by product format (swatch, bullet, liquid)

**10. [Validation & Test Set Strategy](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/10_validation_test_set_strategy.ipynb)** 
- Build validation and test sets from images held out of the training queue
- Proportional sampling to reflect the production distribution, with sufficient per-class coverage for meaningful subgroup evaluation
- Ensures accuracy, IoU, and ΔE metrics are reported on images that never participated in training, model selection, or active learning
