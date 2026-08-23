# CLAUDE.md

ML pipeline that extracts CIELAB colors from ~9k lipstick product images and powers
the search app at lipstickbycolor.github.io (app code lives in the separate
LipstickByColor GitHub org, not here). The main README is the portfolio write-up;
keep its claims in sync with what the notebooks actually do.

## Environment

- Python venv at `.venv/` (Python 3.9). Run everything with `.venv/bin/python` —
  torch, segmentation_models_pytorch, and label-studio-converter live there.
- Device selection in notebooks is `mps` → `cuda` → `cpu` (developed on macOS).

## Pipeline and data contract

Notebooks run in numeric order; each stage's outputs feed the next:

| Notebook | Reads | Writes |
|---|---|---|
| 01 data engineering | `data/product_metadata/product_lipstick_metadata.csv` (raw input; repo ships only the `_sample` version) | `data/processed/products_with_images.csv` |
| 02 data validation | `products_with_images.csv` | same file cleaned (removed rows archived to `invalid_images.csv`), `data/img/original_clean/` |
| 03 training set strategy | `products_with_images.csv` | `data/annotation_sample/s1_*.csv`, `s2_*.csv`, `s3_*.csv` |
| 04 color annotation | manually cropped swatches in `data/img/groundtruth/` | ground-truth CIELAB back into `products_with_images.csv` |
| 05 annotation prep | annotation CSVs, Label Studio JSON exports | `data/processed/annotations*.csv` |
| 06 models | `annotations_combined.csv`, masks | `models/*.pth`, `data/annotations/labels.csv`, active-learning queues |
| 07 strategy comparison | `labels.csv`, checkpoints | analysis only |
| 08 production inference | all images, checkpoints | `data/processed/products_pipeline.csv` |
| 09 visualization | `products_pipeline.csv` | plots only |
| 11 validation/test sets | `products_with_images.csv` | eval set CSVs |

Label Studio JSON exports in `data/processed/` are manual artifacts (exported by
hand from the Label Studio UI) — no notebook produces them. Most of `data/` and
all images are gitignored.

## Conventions (do not regress these)

- **Class labels**: `bullet`, `closed`, `lips`, `liquid`, `pencil`, `swatch`,
  `unclassifiable`. All seven are trained classes read from
  `annotations_combined.csv` — none are confidence-threshold-only. `unclassifiable`
  merges two dead-end annotation labels (`color_not_shown`: sealed container,
  nothing visible; `multi_impage`, a Label Studio typo for "multi image": several
  products/a palette/not a single product shot) since both get the same downstream
  treatment (no color extraction) — do this merge in notebook 05 when loading the
  raw Label Studio export, not by keeping them as separate trained classes. Training
  `unclassifiable` directly (rather than relying on low classifier confidence to
  catch it) was a deliberate choice: confidence isn't a reliable proxy for
  out-of-distribution — a multi-product photo can contain something that looks
  enough like `bullet` to get a confident, wrong prediction. `CLF_THRESHOLD = 0.6`
  still exists as a secondary safety net for images that don't even resemble the
  `unclassifiable` training examples. The old names (`bullet_lipstick`,
  `liquid_lipstick`, `other`, and the pre-merge `color_not_shown`) are deprecated.
  `pencil` and `lips` have no dedicated segmenter data floor yet — check
  `ckpt['classes']` and each class's example count before assuming a class is
  well-supported.
- **Class order comes from the checkpoint**: notebook 06 saves
  `{'model_state_dict': ..., 'classes': [...]}`. Consumers must read
  `ckpt['classes']` — never hardcode the list. A hardcoded list that drifts from
  the checkpoint silently misroutes predictions (this happened: liquid images
  went to the closed-container segmenter).
- **Strategy naming**: the two extraction approaches are the **k-means strategy**
  and the **segmentation strategy**. Never "Model A" / "Model C" — those names
  were deliberately removed.
- **Metric**: Delta E CIE 2000 against ground truth; JND threshold cited as 2.3.
  Notebook 04's pairwise-coverage number is plain Euclidean (ΔE76) — label it as
  such if referenced.

## Editing notebooks

- Preserve cell outputs — especially the cluster grid images in notebook 03
  (~27MB). Never clear outputs to save space without being asked.
- For large notebooks, don't read/rewrite the whole file: extract sources with
  `jq` for review, edit the `.ipynb` JSON with a targeted Python script, and
  verify edited code cells still parse with `ast.parse`.
- Notebooks must run top-to-bottom: define names before the first cell that uses
  them; no duplicate function definitions across cells.
- After editing notebooks and before committing, run the lint:

  ```bash
  .venv/bin/python .claude/skills/check-notebooks/check_notebooks.py
  ```

  (also available as the `check-notebooks` skill). It checks definition order,
  checkpoint class-order drift, the CSV producer chain, and stale references.

## Git

- Commit as `7596706+ConstanzaSchibber@users.noreply.github.com`.
- No `Co-Authored-By` trailers in commit messages.
- Commit only when asked. `data/`, images, and models are largely gitignored —
  don't force-add them.
