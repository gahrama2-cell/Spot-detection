"""
speck_report_mm2.py
───────────────────
Adds width_px & height_px, computes sheet pixel area as a circle whose
diameter = average(width, height).
"""

from pathlib import Path
import json, csv, cv2, numpy as np, math, re

# ── constants ──────────────────────────────────────────────────────────────
D_MM = 164.4                                    # true sheet diameter (mm)
A_SHEET_MM2 = math.pi * (D_MM / 2) ** 2         # true sheet area (mm²)

folder      = Path("Area")
crop_dir    = folder / "cropped_rois"
mask_dir    = folder / "masked_specks"
out_csv     = "speck_report_averageheightwidth_mm2_cropped_images.csv"

crop_dir.mkdir(exist_ok=True)
mask_dir.mkdir(exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────
def first_4_components(stem: str) -> str:
    parts = re.split(r"[ _]+", stem, maxsplit=4)
    return "_".join(parts[:4]).lower()

def mask_dark_speck(roi_bgr):
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return m  # uint8 mask

# ── index JSON files by key (first 4 parts) ────────────────────────────────
json_index = {}
for j in folder.glob("*.json"):
    key = first_4_components(j.stem)
    json_index.setdefault(key, []).append(j)

# ── process each image ─────────────────────────────────────────────────────
rows = []

for img_path in folder.iterdir():
    if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        continue

    key = first_4_components(img_path.stem)
    if key not in json_index:
        print(f"[skip] {img_path.name} → no matching JSON (key={key})")
        continue
    json_path = json_index[key][0]

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[warn] cannot read {img_path.name}")
        continue

    H, W = img.shape[:2]
    D_px = (W + H) / 2                       # diameter in pixels
    sheet_px = math.pi * (D_px / 2) ** 2     # area of that circle (px²)
    mm2_per_px = A_SHEET_MM2 / sheet_px

    with open(json_path) as f:
        ann = json.load(f)

    rect_counter = 0
    for shp in ann["shapes"]:
        if (
            shp.get("shape_type") != "rectangle"
            or not shp["label"]
            or shp["label"][0] not in ("S", "p", "r")
        ):
            continue

        rect_counter += 1
        label = shp["label"]

        (x1, y1) = map(int, shp["points"][0])
        (x2, y2) = map(int, shp["points"][1])
        x1, x2 = sorted((max(0, x1), min(W, x2)))
        y1, y2 = sorted((max(0, y1), min(H, y2)))

        roi      = img[y1:y2, x1:x2]
        roi_px   = roi.size // 3              # rows*cols
        roi_mm2  = roi_px * mm2_per_px

        mask     = mask_dark_speck(roi)
        dark_px  = int(np.count_nonzero(mask))
        dark_mm2 = dark_px * mm2_per_px

        base = f"{img_path.stem}_{label}_{rect_counter}"
        cv2.imwrite(str(crop_dir / f"{base}.png"), roi)
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
        cv2.imwrite(str(mask_dir / f"{base}_mask.png"), masked_roi)

        rows.append([
            img_path.name,
            label,
            W, H,                     # width_px, height_px
            round(sheet_px, 0),
            round(A_SHEET_MM2, 2),
            roi_px,
            round(roi_mm2, 2),
            dark_px,
            round(dark_mm2, 2),
        ])

# ── CSV ────────────────────────────────────────────────────────────────────
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image", "label",
        "width_px", "height_px",
        "sheet_px", "sheet_mm2",
        "roi_px",   "roi_mm2",
        "dark_px",  "dark_mm2"
    ])
    writer.writerows(rows)

print(f"""✓ Done.
  • Crops → {crop_dir}
  • Masks → {mask_dir}
  • CSV   → {out_csv}""")
