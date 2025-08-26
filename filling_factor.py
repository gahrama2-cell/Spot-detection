# area_counter_red.py
from pathlib import Path
import cv2
import numpy as np
import csv

folder  = Path("dark_cropped")                 # run the script inside your image folder
out_csv = "areas.csv"

def red_mask(bgr):
    """Return a binary mask where red pixels are 255."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # two HSV bands catch reds that wrap around 0°
    lower1 = np.array([  0,  50,  50])
    upper1 = np.array([ 10, 255, 255])
    lower2 = np.array([170,  50,  50])
    upper2 = np.array([180, 255, 255])

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    return cv2.bitwise_or(m1, m2)     # uint8 mask: 255 = speck

records = []

for crop in folder.glob("*.jpg"):
    if crop.name.startswith("overlay_"):      # skip the overlays themselves
        continue

    mask_path = folder / f"overlay_{crop.name}.png"
    if not mask_path.exists():
        print(f"[skip] {mask_path.name} missing")
        continue

    roi   = cv2.imread(str(crop))                        # colour
    over  = cv2.imread(str(mask_path), cv2.IMREAD_COLOR) # colour (ignore alpha)

    if roi is None or over is None:
        print(f"[warn] could not read {crop.name}")
        continue
    if roi.shape[:2] != over.shape[:2]:
        raise ValueError(f"size mismatch: {crop.name}")

    mask = red_mask(over)
    dark_px = int(np.count_nonzero(mask))        # red pixels only
    roi_px  = roi.shape[0] * roi.shape[1]

    records.append((crop.name, dark_px, roi_px))

# ---- write CSV ------------------------------------------------------------
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "dark_px", "roi_px", "dark_px/roi_px"])
    for name, dark, roi in records:
        w.writerow([name, dark, roi, dark / roi])


