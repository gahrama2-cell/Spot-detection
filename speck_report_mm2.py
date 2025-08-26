"""
speck_report_mm2.py
───────────────────
Adds real‑area (mm²) columns to the pixel report, assuming every circular
handsheet has a true diameter of 16.44 cm (164.4 mm).
"""

from pathlib import Path
import json, csv, cv2, numpy as np, math

# ── constants ──────────────────────────────────────────────────────────────
D_MM = 164.4                                    # real diameter of each sheet
A_SHEET_MM2 = math.pi * (D_MM / 2) ** 2         # real area of a sheet (mm²)

folder  = Path("Area")                             # current directory
out_csv = "speck_report_mm2.csv"
rows    = []                                    # rows for the CSV

# ── helper: make mask of dark speck inside ROI ─────────────────────────────
def mask_dark_speck(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    return mask

# ── process every image ────────────────────────────────────────────────────
for img_path in folder.iterdir():
    if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        continue

    json_matches = list(folder.glob(f"{img_path.stem}*.json"))
    if not json_matches:
        print(f"[skip] {img_path.name} → no matching JSON")
        continue
    json_path = json_matches[0]

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[warn] cannot read {img_path.name}")
        continue

    H, W   = img.shape[:2]
    img_px = H * W
    mm2_per_px = A_SHEET_MM2 / img_px          # scale factor for this image
    img_mm2 = A_SHEET_MM2                      # by definition

    with open(json_path) as f:
        ann = json.load(f)

    for shp in ann["shapes"]:
        if (
            shp.get("shape_type") != "rectangle"
            or not shp["label"]
            or shp["label"][0] not in ("S", "p", "r")
        ):
            continue

        # corners
        (x1, y1) = map(int, shp["points"][0])
        (x2, y2) = map(int, shp["points"][1])
        x1, x2 = sorted((max(0, x1), min(W, x2)))
        y1, y2 = sorted((max(0, y1), min(H, y2)))

        roi      = img[y1:y2, x1:x2]
        roi_px   = roi.shape[0] * roi.shape[1]
        roi_mm2  = roi_px * mm2_per_px

        mask     = mask_dark_speck(roi)
        dark_px  = int(np.count_nonzero(mask))
        dark_mm2 = dark_px * mm2_per_px

        rows.append([
            img_path.name,
            shp["label"],
            img_px,  round(img_mm2,  2),
            roi_px,  round(roi_mm2,  2),
            dark_px, round(dark_mm2, 2),
        ])

# ── write CSV ──────────────────────────────────────────────────────────────
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image", "label",
        "img_px",  "img_mm2",
        "roi_px",  "roi_mm2",
        "dark_px", "dark_mm2"
    ])
    writer.writerows(rows)

print(f"✓ Done. Report (pixels + mm²) saved as {out_csv}")
