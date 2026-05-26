# Project Priorities

## Architecture

Two parallel tracks:

**Track 1 — Production pipeline**
Image → 2C classifier (swatch vs. container type) → Clustering with type-aware preprocessing → confidence check → LLM fallback for hard cases only

**Track 2 — Research comparison**
Run ML, Claude (full), and mini LLMs on 209 ground truth images → ΔE comparison table

---

## Notebook Status

| Notebook | Status |
|---|---|
| 1A Data Engineering | Complete |
| 1B Data Validation | Complete |
| 2A Sampling | Complete |
| 2B Ground Truth | Complete |
| 2C Image Recognition | Barely started (Label Studio install only) |
| 3 Clustering v2 | Scaled to all images, missing GT evaluation |
| 4 LLM | Outdated model, outdated API, needs re-run |
| 5 Comparison | Cannot run — depends on updated 4 |
| 6 Streamlit | Colab-only, reads from Google Drive |

---

## P1 — 2C: Image Recognition Pipeline

The 2C model feeds Track 1. Design decisions need to be locked before annotating anything, because annotation is labor-intensive and hard to redo.

### Design decisions (lock these first)

- **Image-level labels**: `swatch`, `container_lipstick`, `container_lipgloss`, `container_lipliner`
  - Fast to annotate (one click per image), trains a classifier
  - Tells the pipeline *how* to treat the image
- **Region annotations**: bounding box around the color area within container images
  - Slow to annotate, tells the pipeline *where* the color is
  - Either trains an object detector, or informs a fixed-crop heuristic per image type
- **Key question before annotating**: is the color region location unpredictable enough to justify a detector?
  For lipstick containers, the bullet is almost always at the top center — a fixed crop heuristic may be sufficient, saving significant annotation work.

### Tasks

- [ ] Lock annotation strategy (labels + whether to do region annotations or heuristic crops)
- [ ] Set up Label Studio project with correct label schema
- [ ] Annotate images (manual work — can be done in any session, any amount of time)
- [ ] Train image classifier (ResNet fine-tuning, ~100–200 examples per class)
- [ ] Implement crop strategy for each container type (heuristic or object detector)
- [ ] Integrate 2C output into notebook 3 preprocessing

---

## P2 — Notebook 3: Add Ground Truth Evaluation

The v2 notebook already processed all 9,167 images. It's missing the evaluation step.

- [ ] Filter `products_with_images.csv` to `color_swatch == 1` (209 GT images)
- [ ] Compare `mean_lab` and `peak_lab` predictions against `ground_truth_CIELAB`
- [ ] Compute ΔE per image, report mean/median/distribution
- [ ] Break down ΔE by image type once 2C labels exist

**Time: ~30 min**

---

## P3 — Notebook 4: LLM Comparison (Track 2)

Run on 209 GT images only. Do not run on full dataset until pipeline strategy is finalized (cost).

### Updates needed
- [ ] Replace raw HTTP calls with Anthropic SDK
- [ ] Replace deprecated `claude-3-opus-20240229` with current model
- [ ] Fix input file reference (`products_clustered_final.csv` → `products_with_images.csv` filtered to GT)
- [ ] Run Claude Sonnet on 209 GT images → compute ΔE

### Mini LLM comparison
Run all candidates on the same 209 GT images. Total cost across all providers: ~$2–5.

| Model | Provider | Notes |
|---|---|---|
| Claude Haiku 4.5 | Anthropic | Already in stack, ~10x cheaper than Sonnet |
| GPT-4o mini | OpenAI | Very cheap, strong at structured output |
| Gemini 2.0/2.5 Flash | Google | Competitive pricing, good vision |
| LLaMA 3.2 Vision 11B | Meta (local via Ollama) | Free, may need more prompt tuning |

- [ ] Add each model as a separate function/section in notebook 4
- [ ] Standardize output parsing (all models must return a parseable CIELAB list)
- [ ] Compute ΔE for each model on the 209 GT images

---

## P4 — Notebook 5: Full Comparison

Depends on P2 and P3 being complete.

- [ ] ΔE comparison table: Clustering vs. Claude Sonnet vs. mini LLMs
- [ ] Define "hard case" signal from clustering output (e.g., high cluster variance, large black/white-filtered fraction)
- [ ] Show which images clustering struggles with and whether LLM fills the gap
- [ ] Assess whether image type (from 2C) predicts difficulty

---

## P5 — Hybrid Pipeline (Track 1 completion)

After P2–P4 establish which images ML handles poorly:

- [ ] Define confidence threshold for routing to LLM
- [ ] Run LLM only on hard cases in full 9,167-image dataset (use Batch API — 50% cheaper, async)
- [ ] Consider Haiku if ΔE difference vs. Sonnet is small on hard cases
- [ ] Produce final `products_app.csv` with ML predictions for most, LLM for hard cases

---

## P6 — Polish

These can fill short sessions or be done incrementally.

- [ ] **Notebook 3**: add ΔE by category breakdown to match original notebook
- [ ] **Notebook 2B**: add annotation methodology note (tool used, criteria for valid swatch crop)
- [ ] **README**: fill in `<insert key descriptive stats>` in Data Collection section
- [ ] **experiments.txt — scrape missing image URLs**: ~971 rows have `product_url` but no `img_url`; fetch `og:image` from each page (Amazon links may fail)
- [ ] **experiments.txt — investigate `no_color_info`**: 31.8% of products have no color metadata — check if missingness is systematic by brand or retailer (potential bias)
- [ ] **experiments.txt — fold 'other' category**: 4 products (nectar, heather, dijon mustard, wild azalea) — map to nearest color group
- [ ] **Notebook 6**: update Streamlit notebook to read local files instead of Google Drive, remove Colab deployment instructions

---

## P7 — Standalone Pipeline

Once all stages are stable, extract the core logic from notebooks into a clean, code-only pipeline. Notebooks stay as documentation; the pipeline folder becomes the production-ready version.

```
pipeline/
  run_pipeline.py       # runs all stages in order, stops if a stage fails
  stage_1_validate.py   # logic from 1A + 1B (download, validate, normalize)
  stage_2_sample.py     # logic from 2A + 2B (sampling, ground truth extraction)
  stage_3_cluster.py    # logic from 3 (clustering + GT evaluation)
  stage_4_llm.py        # logic from 4 (LLM color extraction)
  stage_5_compare.py    # logic from 5 (comparison + hard-case routing)
```

- [ ] Extract pipeline once all notebooks are stable and re-run end-to-end
- [ ] Each stage reads a named input file and writes a named output file
- [ ] `run_pipeline.py` checks each stage output exists before running the next

---

## By Time Available

**15–30 min**
- Annotate images in Label Studio (any number, any time)
- Notebook 3: add GT evaluation
- README descriptive stats
- Notebook 2B: annotation methodology note

**1–2 hours**
- Notebook 4: update API + SDK, run Claude Sonnet on GT images
- Add one mini LLM to notebook 4

**Half day**
- Complete mini LLM comparison across all 4 candidates
- Notebook 5: updated comparison

**Multiple sessions**
- 2C annotation (manual, open-ended)
- Train 2C classifier
- Integrate 2C into clustering pipeline
- Full hybrid pipeline

**Last — when everything is stable**
- Standalone pipeline (`pipeline/` folder, `run_pipeline.py`)
