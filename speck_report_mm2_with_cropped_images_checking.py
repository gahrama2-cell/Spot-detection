"""
speck_report_mm2.py
───────────────────
For each handsheet image (+ its LabelMe JSON) this script:
  • Calculates pixel and real‑area (mm²) metrics for the sheet, every ROI
    rectangle (labels starting with S / p / r), and the masked dark speck.
  • Saves each ROI crop to   ./cropped_rois/<base>.png
  • Saves each masked speck to ./masked_specks/<base>_mask.png
  • Appends all data to speck_report_mm2_with_cropped_images.csv
"""

from pathlib import Path
import json, csv, cv2, numpy as np, math
import pandas as pd
import os
from tqdm import tqdm

# ── constants ──────────────────────────────────────────────────────────────
D_MM = 164.4                                    # true sheet diameter (mm)
A_SHEET_MM2 = math.pi * (D_MM / 2) ** 2         # real sheet area (mm²)

folder       = Path("Area")                        # images + JSON here
crop_dir     = folder / "cropped_rois"
mask_dir     = folder / "masked_specks"
out_csv      = "speck_report_mm2_with_cropped_images.csv"

crop_dir.mkdir(exist_ok=True)
mask_dir.mkdir(exist_ok=True)
COLOR      = (0, 0, 255)    # BGR → red; change to (0,255,255)=yellow etc.
ALPHA      = 0.5  
rows = []                                       # CSV rows

# ── helper: create binary mask of dark speck in ROI ────────────────────────
def mask_dark_speck(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    return mask  # uint8: 255 = speck, 0 = background

def finetuning(gt_data, d, sigmaColor, sigmaSpace):
    imgs_names = gt_data['image'].tolist()
# ── iterate over every image file ──────────────────────────────────────────
    for img_path in folder.iterdir():
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        # find any JSON whose name begins with the image’s stem
        json_matches = list(folder.glob(f"{img_path.stem}*.json"))
        if not json_matches:
            print(f"[skip] {img_path.name} → no JSON")
            continue
        json_path = json_matches[0]

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] cannot read {img_path.name}")
            continue

        H, W   = img.shape[:2]
        img_px = H * W
        # img_px = math.pi*((H * W)/2)^2/4
        mm2_per_px = A_SHEET_MM2 / img_px          # scale factor for this image

        with open(json_path) as f:
            ann = json.load(f)

        # index rectangles so filenames stay unique even if labels repeat
        rect_counter = 0

        for shp in ann["shapes"]:
            if (
                shp.get("shape_type") != "rectangle"
                or not shp["label"]
                or shp["label"][0] not in ("S", "p", "r", '1')
            ):
                continue
            rect_counter += 1
            label = shp["label"]
            base_name = f"{img_path.stem}_{label}_{rect_counter}"
            if f"{base_name}.png" not in imgs_names:
                continue
            # rectangle corners
            (x1, y1) = map(int, shp["points"][0])
            (x2, y2) = map(int, shp["points"][1])
            x1, x2 = sorted((max(0, x1), min(W, x2)))
            y1, y2 = sorted((max(0, y1), min(H, y2)))

            roi      = img[y1:y2, x1:x2]
            roi_px   = roi.shape[0] * roi.shape[1]
            roi_mm2  = roi_px * mm2_per_px 
            # roi = cv2.GaussianBlur(roi, (3, 3), 0)
            roi = cv2.bilateralFilter(roi,              # input image
                              d=d,               # pixel‑neighbourhood diameter
                              sigmaColor=sigmaColor,     # larger ⇒ stronger colour smoothing
                              sigmaSpace=sigmaSpace)
            mask = mask_dark_speck(roi)
            k = np.ones((5, 5), np.uint8)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, 2)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 2)
            # num, lbl, stats, _ = cv2.connectedComponentsWithStats(mask)
            # if num > 1:
            #     largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            #     mask = np.where(lbl == largest, 255, 0).astype("uint8")
            dark_px  = int(np.count_nonzero(mask))
            dark_mm2 = dark_px * mm2_per_px
            gt_data.loc[gt_data['image'] == f"{base_name}.png", "dark_px_by auto_mm2" ] = dark_mm2
            # ── save ROI and masked speck images ───────────────────────────────
            
            cv2.imwrite(str(crop_dir / f"{base_name}.png"), roi)

            # masked image: keep only speck pixels, black elsewhere
            masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
            overlay = roi.copy()
            overlay[mask == 255] = COLOR                    # paint cluster solid colour
            viz = cv2.addWeighted(overlay, ALPHA, roi, 1-ALPHA, 0)  # blend
            cv2.imwrite(str(mask_dir / f"{base_name}_mask.png"), viz)

            # ── collect CSV row ────────────────────────────────────────────────
            rows.append([
                img_path.name,
                label,
                img_px,
                round(A_SHEET_MM2, 2),
                roi_px,
                round(roi_mm2, 2),
                dark_px,
                round(dark_mm2, 2),
            ])
    return rows, gt_data
    # ── write the CSV ──────────────────────────────────────────────────────────
   
if __name__=='__main__':
    gt = pd.read_csv("GT_checking.csv")
    # mm2 = gt['dark_px_by hand_mm2'].tolist()
    gt['dark_px_by auto_mm2'] = 0.
    ds = [i for i in range(5, 35, 5)]
    sigmaColors = [i for i in range(5, 45, 5)]
    sigmaSpaces = [i for i in range(5, 35, 5)]

    best_d = -1
    best_sigc = -1
    best_sigs = -1
    best_rows = []
    least_error = 10.7306852
    for d1 in tqdm(ds):
        for sigc in tqdm(sigmaColors):
            for sigs in tqdm(sigmaSpaces):
                print(f"Current params: d={d1}, sigmaColor={sigc}, sigmaSpace={sigs}")
                rows, gt_data = finetuning(gt, d1, sigc, sigs)
                gt['error'] = np.abs(gt['dark_px_by hand_mm2'] - gt['dark_px_by auto_mm2'])/gt['dark_px_by hand_mm2'] * 100
                if least_error > gt['error'].std():
                    least_error = gt['error'].std()
                    best_d = d1
                    best_sigc = sigc
                    best_sigs = sigs
                    best_rows = rows
                    print(f"Best STD: {least_error}\nBest params: d={best_d}, sigmaColor={best_sigc}, sigmaSpace={best_sigs}")
                else:
                    print(f"Best STD: {least_error}\nBest params: d={best_d}, sigmaColor={best_sigc}, sigmaSpace={best_sigs}")


    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "label",
            "img_px",  "img_mm2",
            "roi_px",  "roi_mm2",
            "dark_px", "dark_mm2"
        ])
        writer.writerows(best_rows)
    
    print(f"Best STD: {least_error}\nBest params: d={best_d}, sigmaColor={best_sigc}, sigmaSpace={best_sigs}")

