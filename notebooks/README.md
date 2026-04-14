# Notebooks

This folder contains the notebooks for the project, organized in execution order.

---

## Setup

**1. Install dependencies:**
```
pip install -r requirements.txt
```

**2. Set up your API key** (required for notebook 4 only):

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_actual_key_here
```

**3. Add data files:**

Place the following in `data/processed/` before running notebooks 3–6:
- `products_with_images.csv` (output of notebook 1, updated by 2A and 2B)
- `annotation_sample.csv` (output of notebook 2A)
- `ground_truth_labels.csv` (output of notebook 2B)
- `products_clustered_v2.csv`, `products_clustered_final.csv` (outputs of notebook 3)
- `products_llm_final.csv` (output of notebook 4)

Images should be placed in `data/img/original/`.

---

## Notebooks

**1. [Data Engineering](https://github.com/ConstanzaSchibber/capstone_colors/blob/main/notebooks/1_DataEngineering.ipynb)**
- Validate URLs and download product images from makeup retailers
- Check that all downloaded files are valid, uncorrupted images
- Exploratory data analysis: product categories, image size, resolution, color distributions

**2A. [Data Annotation: Sampling](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/2A_DataAnnotatioSampling.ipynb)**
- Consolidate 200+ raw `parent_color` values into 18 color groups via a keyword-based taxonomy
- Calculate sample size using Cochran's formula (n=188, rounded to 200) with CIELAB L* std from a prior lipstick study
- Select 222 images via stratified proportional sampling with a minimum floor of 5 per group
- Copy sampled images to `data/img/groundtruth/` for manual annotation

**2B. [Data Annotation: Ground Truth Labeling](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/2B_DataAnnotationGT.ipynb)**
- Identify which sampled images received a manual crop (209 out of 222)
- Extract mean CIELAB color from each cropped swatch image as the ground truth label
- Store ground truth values in metadata for use in model evaluation
- Visualize ground truth colors as a swatch grid grouped by color taxonomy
- Assess color space coverage: mean pairwise ΔE = 30.54 across the sample

**3. [Method A: Color Segmentation with Clustering](https://github.com/ConstanzaSchibber/capstone_colors/blob/main/notebooks/3_Model_A_Clustering.ipynb)**
- Apply k-means clustering to extract dominant colors from makeup images
- Round 1: baseline algorithm on training set, validated with Delta E
- Round 2: refine by filtering out near-black and near-white clusters (background/packaging)
- Round 3: improve handling of PNG images and CIELAB conversion
- Final evaluation on test set: strong results for blush and lipgloss; lipliner needs further refinement

**4. [Method B: Color Identification with Multimodal LLM](https://github.com/ConstanzaSchibber/capstone_colors/blob/main/notebooks/4_Model_B_LLM.ipynb)**
- Use Claude (multimodal LLM) to identify CIELAB colors directly from product images
- Round 1: baseline prompt, scaled to full dataset
- Round 2: refined category-specific prompts — mean Delta E improved from 16.4 to 11.5 (30% improvement)
- Final evaluation: 70% of cases have Delta E below 20, 50% below 15

> **Requires an Anthropic API key.** See setup instructions above.

**5. [Comparison of Methods A & B](https://github.com/ConstanzaSchibber/capstone_colors/blob/main/notebooks/5_Comparison.ipynb)**
- Compare Delta E results across both methods for each product category
- LLM outperforms clustering overall, particularly for lipstick (62.83% of cases)
- Includes visualization of color predictions vs. ground truth

**6. [Streamlit App](https://github.com/ConstanzaSchibber/capstone_colors/blob/main/notebooks/6_StreamlitApp.ipynb)**
- Generate color swatches for the UI
- Streamlit app code for filtering makeup by color, brand, and category
- Deployment instructions

> **Just want to run the app?** A self-contained version is available at [github.com/ConstanzaSchibber/makeup-filter](https://github.com/ConstanzaSchibber/makeup-filter).
