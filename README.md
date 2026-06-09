# Project: Advancing Lipstick Color Matching with ML and Multimodal LLM

**tl;dr:**

- **Goal:** Identify the color of lipstick products in CIELAB color space to enable comparison by standardized shade rather than by the creative names brands assign.
- **Data:** Product metadata and images collected from makeup retailers via API and web scraping.
- **Methods:** (1) Color segmentation with clustering, (2) Multimodal Large Language Model (Claude)
- **App:** Web interface for filtering lipstick by color
- **Tech stack:** Python, Jupyter, Streamlit

*Table of Contents*
- [Problem & Solution](#problem--solution)
- [Data Collection](#data-collection)
- [Human Annotation](#human-annotation)
- [Method 1: Color Segmentation](#method-1-color-segmentation)
- [Method 2: Multimodal LLM](#method-2-multimodal-llm)
- [Comparative Analysis](#comparative-analysis)
- [Streamlit App](#streamlit-app)

---

## Problem & Solution

Lipstick products have fanciful and inconsistent color names, making it difficult for consumers to find or compare shades across brands. For example, the following products are all labeled "mauve," yet they are clearly different shades — some are not even mauve by any standard definition.

<div align="center">
<img src="https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/5be3cc59ebc4906092fa95ccbdc54b890ca8827e/img/Screen%20Shot%202024-08-24%20at%208.52.52%20PM.png" width=50%>
</div>

Retailer color filters are also limited. Below are the lipstick color options at Sephora and Ulta, compared to the more granular palette my app provides:

<div align="center">
<img src="https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/img/filters_retail.png" width=75%>
</div>

By mapping lipstick colors to the [CIELAB color space](https://en.wikipedia.org/wiki/CIELAB_color_space), I create a standardized, perceptually uniform representation that enables accurate shade comparison across brands. CIELAB represents color in three dimensions: L (lightness), a (green to red), and b (blue to yellow). Equal numerical differences in CIELAB correspond to roughly equal perceived differences to the human eye, making it ideal for color matching.

---

## Data Collection

Product metadata, including brand, product name, shade, and color descriptors, was collected from retailer APIs (Ulta, Sephora) and through web scraping of individual brand websites. Each record was matched to a product image URL, which was then downloaded and validated.

`<insert key descriptive stats>`

> **Note on data availability:** The full metadata file (`data/product_metadata/product_lipstick_metadata.csv`) is not committed to this repository to protect the scraped dataset from being reused wholesale. A sample of 173 rows is included at `data/product_metadata/product_lipstick_metadata_sample.csv` so the notebooks can be run end-to-end. If you need access to the full dataset for research purposes, please open an issue.

---

## Human Annotation

To create ground truth color labels, I selected a stratified sample of 222 lipstick images across 18 color groups using Cochran's formula for sample size (n=188, rounded to 200). Each sampled image was manually cropped to isolate the lipstick color swatch, and the mean CIELAB color was extracted from the cropped region as the ground truth value. Of the 222 sampled images, 209 yielded a valid ground truth color. See [notebook 2A](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/2A_DataAnnotatioSampling.ipynb) (sampling) and [notebook 2B](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/2B_DataAnnotationGT.ipynb) (ground truth extraction).

---

## Method 1: Color Segmentation

<details>
<summary>Show details</summary>

I developed a method to identify CIELAB color shades in lipstick images using k-means clustering (see [notebook](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/3_Model_A_Clustering.ipynb)).

**Key steps:**
1. **Initial exploration:** Applied k-means clustering to test images to extract dominant colors and establish a baseline approach.
2. **Scaling:** Expanded to the full dataset with training, validation, and test splits. Evaluated using [Delta E (ΔE)](https://en.wikipedia.org/wiki/Color_difference), which measures perceptual color difference.
3. **Refinement:** Filtered out near-black and near-white clusters to reduce interference from packaging and backgrounds, significantly improving accuracy.
4. **Final evaluation:** Tested on held-out data to assess generalization.

**Results:** About 62% of test images had a ΔE below 15. No images exceeded ΔE 45 and 95% were below 30. Mean ΔE was lower for blush (10.5) and lipgloss (14.02) than for lipliner (17.8) and lipstick (18.3).

<div align="center">
<img src=https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/img/deltaE.png width=70%>
</div>

</details>

---

## Method 2: Multimodal LLM

<details>
<summary>Show details</summary>

I used `Claude`, a multimodal large language model, to identify CIELAB color values directly from product images (see [notebook](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/4_Model_B_LLM.ipynb)).

**Key steps:**
1. **Initial implementation:** Built a pipeline to process images and estimate CIELAB colors via Claude API.
2. **Prompt engineering:** Refined prompts over multiple rounds, tailoring them by product category.
3. **Evaluation:** Used ΔE to compare predictions against ground truth.

**Results:**
- Mean ΔE improved from 16.4 to 11.5 after prompt refinement (30% improvement).
- Median ΔE decreased from 12.8 to 10.56 (17% improvement).
- 70% of cases have ΔE below 20; 50% below 15.
- Eliminated cases with very high ΔE (>40).

</details>

---

## Comparative Analysis

<details>
<summary>Show details</summary>

I [compared both methods](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/5_Comparison.ipynb) using ΔE. A lower ΔE indicates better color accuracy.

| Category | LLM > Clustering (%) | Clustering > LLM (%) |
|----------|----------------------|----------------------|
| blush    | 34.15                | 59.76                |
| lipgloss | 36.89                | 57.38                |
| lipliner | 41.38                | 44.83                |
| lipstick | 28.27                | 62.83                |

The LLM outperformed clustering across all categories, with the largest advantage for lipstick (62.83% of cases). The LLM's ability to incorporate contextual understanding of packaging likely explains its edge. Lipliner was the most balanced category, suggesting clustering holds up reasonably well for thin, high-contrast products.

</details>

---

## Streamlit App

A web app for filtering lipstick by color, brand, and category. Color data is pre-processed using the LLM predictions and grouped by similarity. Users interact with color swatches to filter results. See [notebook](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/6_StreamlitApp.ipynb) or the standalone repo at [github.com/ConstanzaSchibber/makeup-filter](https://github.com/ConstanzaSchibber/makeup-filter).

Before selecting filters:

![img](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/img/myapp.jpg)

After filtering by a color shade:

![img](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/img/appl_filter.png)
