# Notebooks

This folder contains the notebooks for the project, organized in execution order.

> **Just want to see the app?** Try it at [lipstickbycolor.github.io](https://lipstickbycolor.github.io/) — source at [github.com/LipstickByColor](https://github.com/LipstickByColor).

---

## Setup

**1. Install dependencies:**
```
pip install -r requirements.txt
```

**2. Data files by stage:**

| Notebook | Reads | Writes |
|----------|-------|--------|
| 1A | raw metadata CSVs | `products_with_images.csv` |
| 1B | `products_with_images.csv` | cleaned `products_with_images.csv` |
| 2A | `products_with_images.csv` | `annotation_sample.csv` |
| 2B | annotation masks | `ground_truth_labels.csv` |
| 2C | `annotation_sample.csv` | images for Label Studio |
| 3A | annotated labels + images | trained ResNet-18 classifier |
| 3B | classifier + images | evaluated clustering results |
| 4  | classifier + all images | `products_pipeline.csv` |
| 5  | `products_pipeline.csv` | visualizations |

Images should be placed in `data/img/original/`.

---

## Notebooks

**1A. [Data Engineering](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/1A_DataEngineering.ipynb)**
- Collect product metadata from Ulta and Sephora via API and web scraping
- Validate URLs and download product images
- Exploratory data analysis: product categories, image size, resolution, color distributions

> **To run this notebook**, use the sample metadata file included in the repo: `data/product_metadata/product_lipstick_metadata_sample.csv` (173 products). The full scraped dataset is not included.

**1B. [Image Validation](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/1B_DataValidation.ipynb)**
- Check all downloaded images for corruption and readability
- Remove invalid files; output a clean image set for downstream processing

**2A. [Data Annotation: Sampling](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/2A_DataAnnotationSampling.ipynb)**
- Consolidate 200+ raw `parent_color` values into 18 color groups via a keyword-based taxonomy
- Calculate sample size using Cochran's formula (n=188, rounded to 200) with CIELAB L* std from a prior lipstick study
- Select 222 images via stratified proportional sampling with a minimum floor of 5 per group
- Copy sampled images to `data/img/groundtruth/` for manual annotation

**2B. [Data Annotation: Ground Truth Labeling](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/2B_DataAnnotationGT.ipynb)**
- Identify which sampled images received a manual crop (209 out of 222)
- Extract mean CIELAB color from each cropped swatch image as the ground truth label
- Visualize ground truth colors as a swatch grid grouped by color taxonomy
- Assess color space coverage: mean pairwise ΔE = 30.54 across the sample

**2C. [Data Annotation: Image Type Recognition](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/2C_DataAnnotationImageRecognition.ipynb)**
- Prepare annotation sample for labeling in Label Studio
- Classify images by presentation type: `swatch`, `bullet`, `liquid`, `closed`, `color_not_shown`
- Image type determines which color extraction strategy to apply downstream

**3A. [Image Classifier](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/3A_ImageRecognition.ipynb)**
- Fine-tune ResNet-18 to classify product images into four types: `swatch`, `bullet_lipstick`, `liquid_lipstick`, `other`
- Evaluate classifier accuracy on held-out set
- Apply type-conditional color extraction: each product type routes to the appropriate extraction method

**3B. [Color Segmentation with Clustering](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/3B_Clustering.ipynb)**
- K-means clustering to extract dominant colors, routed by the Stage 1 classifier from 3A
- Filters out near-black and near-white clusters (background/packaging noise)
- Evaluate against ground truth labels using Delta E

**4. [Production Pipeline](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/4_Pipeline.ipynb)**
- Apply the full two-stage pipeline to all ~9k product images
- Stage 1: classify each image with the ResNet-18 classifier
- Stage 2: extract color using the type-appropriate method (clustering or swatch extraction)
- Output: `products_pipeline.csv` with L, a, b, and hex values for every product

**5. [CIELAB Visualization](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/5_CIELAB_Visualization.ipynb)**
- Visualize the distribution of extracted colors across the CIELAB color space
- a\*–b\* scatter plot (chromatic plane) and L\* distribution (lightness)
- Faceted views by product format (swatch, bullet, liquid)
