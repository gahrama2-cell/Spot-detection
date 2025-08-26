"""
speck_report_px.py
──────────────────
Creates speck_report_px.csv with, for every rectangle drawn around a dark spot:

    image   – filename of the handsheet photo
    label   – rectangle label (must start with S / p / r)
    img_px  – pixel area of the whole handsheet image
    roi_px  – pixel area of that rectangle
    dark_px – pixel area of the dark speck isolated inside the rectangle
"""

from pathlib import Path
import json, csv, cv2, numpy as np

folder   = Path("Area")                 # current directory
out_csv  = "speck_report_px.csv"
rows     = []                        # accumulate CSV rows here

# ────────────────────────────────────────────────────────────────────────────
def mask_dark_speck(roi_bgr):
    """Binary mask: 255 where the speck is, 0 elsewhere (Otsu + INV)."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    return mask

# ────────────────────────────────────────────────────────────────────────────
for img_path in folder.iterdir():
    if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        continue                                    # skip non‑image files

    # find any JSON whose name begins with the image's stem
    json_matches = list(folder.glob(f"{img_path.stem}*.json"))
    if not json_matches:
        print(f"[skip] {img_path.name} → no matching JSON")
        continue
    json_path = json_matches[0]                    # first match is fine

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[warn] cannot read {img_path.name}")
        continue

    H, W   = img.shape[:2]
    img_px = H * W                                # pixel area of full image

    with open(json_path) as f:
        ann = json.load(f)

    # ── process every rectangle labelled S / p / r ─────────────────────────
    for shp in ann["shapes"]:
        if (
            shp.get("shape_type") != "rectangle"
            or not shp["label"]
            or shp["label"][0] not in ("S", "p", "r")
        ):
            continue

        # extract & clamp rectangle coordinates
        (x1, y1) = map(int, shp["points"][0])     # first corner
        (x2, y2) = map(int, shp["points"][1])     # opposite corner
        x1, x2 = sorted((max(0, x1), min(W, x2)))
        y1, y2 = sorted((max(0, y1), min(H, y2)))

        roi      = img[y1:y2, x1:x2]
        roi_px   = roi.shape[0] * roi.shape[1]

        mask     = mask_dark_speck(roi)
        dark_px  = int(np.count_nonzero(mask))

        rows.append([
            img_path.name,
            shp["label"],
            img_px,
            roi_px,
            dark_px
        ])

# ────────────────────────────────────────────────────────────────────────────
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "label", "img_px", "roi_px", "dark_px"])
    writer.writerows(rows)

print(f"✓ Done. Pixel report saved as {out_csv}")
