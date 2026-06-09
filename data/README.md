This `data/` directory contains all images, annotations, and processed outputs for the lipstick color extraction project.

## Directory Structure

```
data/
├── img/                        # All product images
│   ├── original/               # Raw downloaded images (~9,500 files)
│   ├── original_clean/         # Cleaned originals with corrupted files removed (~9,167 files)
│   ├── groundtruth/            # Manually annotated ground truth images (222 files)
│   ├── groundtruth_old/        # Previous version of ground truth images
│   ├── annotation_sample/      # Images sampled for annotation (208 files)
│   ├── annotation_sample_closed/  # Images used in closed annotation session (40 files)
│   ├── annotation_label_distribution.png
│   ├── annotation_label_distribution_combined.png
│   └── product_type_distribution.png
│
├── annotations/                # Human annotation outputs
│   ├── labels.csv              # Per-image labels with CIELAB ground truth values and mask paths
│   ├── active_learning_queue.csv  # Images queued for annotation via active learning
│   └── masks/                  # Segmentation masks as PNG files (247 files)
│
├── processed/                  # Pipeline outputs and app-ready data
│   ├── README.md
│   ├── metadata.csv            # Original metadata with image download URLs
│   ├── products_with_images.csv   # Products matched to downloaded image files
│   ├── corrupted_images.csv    # Log of images that failed to load or convert
│   ├── annotation_sample.csv   # Sampling frame for annotation
│   ├── annotations.csv         # Processed annotation results
│   ├── annotations_combined.csv   # Annotations merged across sessions
│   ├── annotations_label_studio.json          # Label Studio export (main session)
│   ├── annotations_label_studio_closed_addition.json  # Label Studio export (closed session)
│   ├── lipstick_colors.csv     # Extracted colors per product (brand, product, shade, CIELAB)
│   ├── products_pipeline.csv   # Full pipeline output with color extraction results
│   ├── products_clustered.csv  # Pipeline output with GMM color cluster assignments
│   └── products_app.csv        # Final table powering the Streamlit app
│
└── product_metadata/           # Product-level reference data
    ├── product_lipstick_metadata_sample.csv  # 173-row sample for running notebooks (see note below)
    └── brand_tiers.json        # Brand tier heuristics (luxury, mid-range, drugstore, etc.)
```

## Key Files

| File | Description |
|------|-------------|
| `processed/metadata.csv` | Starting point — raw product metadata scraped from retailer sites, includes image download URLs |
| `processed/products_app.csv` | End product — merged table used by the Streamlit app, includes extracted CIELAB colors and cluster assignments |
| `annotations/labels.csv` | Ground truth labels with CIELAB values; used to evaluate color extraction accuracy |
| `product_metadata/product_lipstick_metadata_sample.csv` | Reference metadata including shade descriptions, skintone/undertone tags, cruelty-free/vegan flags, and pricing |

## Note on Data Availability

The full product metadata file (`product_lipstick_metadata.csv`) is not committed to this repository to protect the scraped dataset from being reused wholesale. A sample of 173 rows — drawn from the annotated ground truth set — is included at `product_metadata/product_lipstick_metadata_sample.csv` so the notebooks can be run end-to-end. If you need access to the full dataset for research purposes, please open an issue.
