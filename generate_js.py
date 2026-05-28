"""
Regenerate lipstick-data.js and lipstick-images.js from products_pipeline.csv.
- Colors (L, a, b, hex) come from the new CSV
- Finish is preserved from the existing lipstick-data.js (joined on brand+product+shade)
- Any product in the CSV without a matching finish defaults to "Cream"
"""

import csv
import re
import json
from pathlib import Path

CSV_PATH   = Path("data/processed/products_pipeline.csv")
JS_IN      = Path("../lipstick_website/lipstick-data.js")
JS_OUT     = Path("../lipstick_website/lipstick-data.js")
IMAGES_OUT = Path("../lipstick_website/lipstick-images.js")

# ── 1. Extract finish lookup from existing JS ─────────────────────────────────
finish_lookup = {}  # (brand, product, shade) -> finish
rows_text = JS_IN.read_text(encoding="utf-8")
# _PRODUCT_ROWS entries look like: ["Brand","Product","Shade","Finish",L,a,b,"#hex"]
for m in re.finditer(r'\["([^"]+)","([^"]+)","([^"]+)","([^"]+)"', rows_text):
    key = (m.group(1).lower(), m.group(2).lower(), m.group(3).lower())
    finish_lookup[key] = m.group(4)

print(f"Loaded {len(finish_lookup)} finish entries from existing JS")

# ── 2. Read new CSV ───────────────────────────────────────────────────────────
products = []
images   = {}
missing_finish = 0

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        brand   = row["brand"].strip()
        product = row["product"].strip()
        shade   = row["shade"].strip()
        if not row["L"] or not row["a"] or not row["b"] or not row["hex_color"]:
            continue
        L       = float(row["L"])
        a       = float(row["a"])
        b       = float(row["b"])
        hex_col = row["hex_color"].strip()
        img_url = row["img_url"].strip()

        key = (brand.lower(), product.lower(), shade.lower())
        finish = finish_lookup.get(key)
        if finish is None:
            finish = "Cream"
            missing_finish += 1

        products.append((brand, product, shade, finish, L, a, b, hex_col))

        if img_url:
            img_key = f"{brand.lower()}|{product.lower()}|{shade.lower()}"
            images[img_key] = img_url

print(f"Loaded {len(products)} products from CSV")
print(f"  {missing_finish} products had no finish match — defaulted to 'Cream'")

# ── 3. Read existing JS up to _PRODUCT_ROWS, keep header (math functions) ─────
header_end_marker = "const _PRODUCT_ROWS"
header = rows_text[:rows_text.index(header_end_marker)]

# Keep everything from REAL_PRODUCTS onward (wheel data, hexToLab, getClosestColors)
tail_start_marker = "\nconst REAL_PRODUCTS"
tail = rows_text[rows_text.index(tail_start_marker):]

# ── 4. Build new _PRODUCT_ROWS block ─────────────────────────────────────────
def fmt_row(brand, product, shade, finish, L, a, b, hex_col):
    return (f'[{json.dumps(brand)},{json.dumps(product)},{json.dumps(shade)},'
            f'{json.dumps(finish)},{round(L,2)},{round(a,2)},{round(b,2)},{json.dumps(hex_col)}]')

rows_js = ",\n".join(fmt_row(*p) for p in products)
product_block = f"const _PRODUCT_ROWS = [\n{rows_js}\n];"

# ── 5. Write new lipstick-data.js ─────────────────────────────────────────────
new_js = header + product_block + tail
JS_OUT.write_text(new_js, encoding="utf-8")
print(f"Written {JS_OUT}")

# ── 6. Write new lipstick-images.js ──────────────────────────────────────────
copyright = "// Copyright (c) 2025 Constanza Schibber. Licensed under CC BY-NC 4.0 — non-commercial use only. https://creativecommons.org/licenses/by-nc/4.0/\n"
images_js = (copyright +
             "// Auto-generated: maps \"brand|product|shade\" (lowercased) -> image URL\n"
             "// Built from products_pipeline.csv\n"
             f"window.LIPSTICK_IMAGES = {json.dumps(images, ensure_ascii=False)};")
IMAGES_OUT.write_text(images_js, encoding="utf-8")
print(f"Written {IMAGES_OUT}")
