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
| 03_annotation_sampling | `products_with_images.csv` | `annotation_sample.csv` |
| 04_annotation_ground_truth | annotation masks | `ground_truth_labels.csv` |
| 05_annotation_image_recognition | `annotation_sample.csv` | images for Label Studio |
| 06_model_image_recognition | annotated labels + images | trained ResNet-18 classifier, U-Net segmenters |
| 07_model_clustering | classifier + images | evaluated clustering results |
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

**03. [Annotation: Sampling](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/03_annotation_sampling.ipynb)**
- Consolidate 200+ raw `parent_color` values into 18 color groups via a keyword-based LLM-assisted taxonomy.
- Calculate sample size using Cochran's formula (n=188, rounded to 200) with CIELAB L* std from a prior lipstick study
- Select 222 images via stratified proportional sampling with a minimum floor of 5 per group
- Copy sampled images to `data/img/groundtruth/` for manual annotation

**04. [Annotation: Ground Truth Labeling](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/04_annotation_ground_truth.ipynb)**
- Identify which sampled images received a manual crop (209 out of 222)
- Extract mean CIELAB color from each cropped swatch image as the ground truth label
- Visualize ground truth colors as a swatch grid grouped by color taxonomy
- Assess color space coverage: mean pairwise ΔE = 30.54 across the sample

**05. [Annotation: Image Type Recognition](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/05_annotation_image_recognition.ipynb)**
- Prepare annotation sample for labeling in Label Studio
- Classify images by presentation type: `swatch`, `bullet`, `liquid`, `closed`, `color_not_shown`
- Image type determines which color extraction strategy to apply downstream

**06. [Model: Image Recognition & Segmentation](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/06_model_image_recognition.ipynb)**
- Fine-tune ResNet-18 to classify product images into four types: `swatch`, `bullet_lipstick`, `liquid_lipstick`, `other`
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
