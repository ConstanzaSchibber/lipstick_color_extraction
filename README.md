# Project: Makeup Color Extraction & Retrieval System

**tl;dr:**

- **Goal:** Identify the color of lipstick products in CIELAB color space to enable comparison by standardized shade rather than by the creative names brands assign.
- **Data:** Product metadata and images collected from makeup retailers via API and web scraping.
- **Methods:** 
- **App:** Web interface for filtering lipstick by color
- **Tech stack:** Python, Jupyter, Streamlit

---

> **Just want to see the app?** Try it at [lipstickbycolor.github.io](https://lipstickbycolor.github.io/). Source code at [github.com/LipstickByColor](https://github.com/LipstickByColor).

> **Just want to read a high-level overview?** [About](https://lipstickbycolor.github.io/about.html).
---

*Table of Contents*
- [Problem & Solution](#problem--solution)
- [Data Collection](#data-collection)
- [Human Annotation](#human-annotation)
- [Method 1: Color Segmentation](#method-1-color-segmentation)
- [Method 2: Multimodal LLM](#method-2-multimodal-llm)
- [Comparative Analysis](#comparative-analysis)
- [Streamlit App](#streamlit-app)
- [Citation](#citation)

---

## Problem & Solution

Lipsticks come in every color and shade imaginable, but finding a specific color or shade online is surprisingly difficult.

<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td width="60%" valign="top" style="border: none;">First, brand naming is inconsistent and opaque. Brands use evocative names like "Velvet Plum," "Midnight Berry," and "Spiced Rosewood" that don't map to a specific color. Even when a color appears in the name, it isn't consistent across or within brands. All of the shades below, for instance, are called "mauve" by different brands.</td>
    <td width="40%" align="center" style="border: none;"><img src="https://raw.githubusercontent.com/ConstanzaSchibber/capstone_colors/b633a98ab61db59259cd5440def7f9f6f10a3267/img/Screen%20Shot%202024-08-24%20at%208.52.52%20PM.png" width="100%"></td>
  </tr>
</table>


Second, retailer search and filtering tools are inadequate. Filtering by "Pink" returns hundreds of results spanning wildly different shades such as the ones below.

<table>
  <tr>
    <td align="center" width="25%"><b>Amazon</b><br><img src="https://raw.githubusercontent.com/LipstickByColor/LipstickByColor.github.io/19fa96b761e73191b184296ba09004a47e716268/assets/amazon-pink-results.png" width="100%"></td>
    <td align="center" width="25%"><b>Google Shopping</b><br><img src="https://raw.githubusercontent.com/LipstickByColor/LipstickByColor.github.io/19fa96b761e73191b184296ba09004a47e716268/assets/google-pink-results.png" width="100%"></td>
    <td align="center" width="25%"><b>Sephora</b><br><img src="https://raw.githubusercontent.com/LipstickByColor/LipstickByColor.github.io/19fa96b761e73191b184296ba09004a47e716268/assets/sephora-pink-results.png" width="60%"></td>
    <td align="center" width="25%"><b>Ulta</b><br><img src="https://raw.githubusercontent.com/LipstickByColor/LipstickByColor.github.io/19fa96b761e73191b184296ba09004a47e716268/assets/ulta-pink-results.png" width="60%"></td>
  </tr>
</table>


Some retailers like Sephora and Ulta offer limited color filters, most likely based on metadata supplied by brands, that collapse the entire spectrum into a handful of broad buckets. Others like Amazon and Google Shopping offer no color filters at all, forcing users to search by name. While "pink" or "red" might feel obvious, shades like mauve, dusty rose, or terracotta are ambiguous. And there's no guarantee the search engine returns useful results.

<table>
  <tr>
    <td align="center" width="50%"><b>Sephora</b><br><img src="https://raw.githubusercontent.com/ConstanzaSchibber/lipstick_color_extraction/main/img/sephorafilter.png" width="100%"></td>
    <td align="center" width="50%"><b>Ulta</b><br><img src="https://raw.githubusercontent.com/ConstanzaSchibber/lipstick_color_extraction/main/img/ultafilter.png" width="100%"></td>
  </tr>
</table>


Given these limitations, not only it's difficult to discover and search for lipsticks, but comparing colors across brands or finding a cheaper option is extremely difficult. 



By mapping lipstick colors to the [CIELAB color space](https://lipstickbycolor.github.io/color-guide.html), I create a standardized, perceptually uniform representation that enables accurate shade comparison across brands. CIELAB represents color in three dimensions: L (lightness), a (green to red), and b (blue to yellow). Equal numerical differences in CIELAB correspond to roughly equal perceived differences to the human eye, making it ideal for color matching.

---

## Data Collection

Product metadata, including brand, product name, shade, and color descriptors, was collected from retailer APIs (Ulta, Sephora) and through web scraping of individual brand websites. Each record was matched to a product image URL, which was then downloaded and validated.

`<insert key descriptive stats>`

> **Note on data availability:** The full metadata file (`data/product_metadata/product_lipstick_metadata.csv`) is not committed to this repository to protect the scraped dataset from being reused wholesale. A sample of 173 rows is included at `data/product_metadata/product_lipstick_metadata_sample.csv` so the notebooks can be run end-to-end. If you need access to the full dataset for research purposes, please open an issue.

---

## Human Annotation

To create ground truth color labels, I selected a stratified sample of 222 lipstick images across 18 color groups using Cochran's formula for sample size (n=188, rounded to 200). Each sampled image was manually cropped to isolate the lipstick color swatch, and the mean CIELAB color was extracted from the cropped region as the ground truth value. Of the 222 sampled images, 209 yielded a valid ground truth color. See [notebook 2A](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/2A_DataAnnotationSampling.ipynb) (sampling) and [notebook 2B](https://github.com/ConstanzaSchibber/lipstick_color_extraction/blob/main/notebooks/2B_DataAnnotationGT.ipynb) (ground truth extraction).

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{schibber2024lipstick,
  author       = {Schibber, Constanza},
  title        = {Advancing Lipstick Color Matching with ML and Multimodal LLM},
  year         = {2024},
  publisher    = {GitHub},
  url          = {https://github.com/ConstanzaSchibber/lipstick_color_extraction},
  note         = {Licensed under CC BY-NC 4.0}
}
```
