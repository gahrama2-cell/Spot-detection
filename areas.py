import cv2
import json
import numpy as np

# 1 · load the rectangle from annotationsMe ------------------------------
with open("annotations.json") as f:
    ann = json.load(f)

rect_pts = next(s for s in ann["shapes"] if s["annotations"] == "Spot")["points"]
(x1, y1), (x2, y2) = map(int, rect_pts)          # rectangle corners

# 2 · load images ---------------------------------------------------
img  = cv2.imread("original.png")                # BGR
h, w = img.shape[:2]

# choose *one* of the two branches below
# ------------------------------------------------------------------
# 2A · if you have a true mask (white = speck) ----------------------
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)[:]   # uint8

# 2B · if you only have a red overlay ------------------------------
# over = cv2.imread("overlay.png")                      # BGR
# hsv   = cv2.cvtColor(over, cv2.COLOR_BGR2HSV)
# m1    = cv2.inRange(hsv, (  0, 50, 50), ( 10,255,255))
# m2    = cv2.inRange(hsv, (170, 50, 50), (180,255,255))
# mask  = cv2.bitwise_or(m1, m2)                        # uint8 mask

# 3 · isolate the rectangle region -------------------------------
roi_mask = mask[y1:y2, x1:x2]
area_speck_px = int(np.count_nonzero(roi_mask))          # pixels

# 4 · optional: areas for context -------------------------------
area_rect_px   = (y2 - y1) * (x2 - x1)        # rectangle itself
area_sheet_px  = h * w                        # whole handsheet (pixels)

coverage_rect  = area_speck_px / area_rect_px
coverage_sheet = area_speck_px / area_sheet_px

# 5 · optional: convert to physical units -------------------------
# If you know the real‑world diameter of the handsheet, D_real_mm,
# and you can measure the diameter in pixels, D_px (e.g. w or h):
D_real_mm = 160      # example: 160 mm handsheet
D_px      = w        # assume the sheet fills the width

px_per_mm = D_px / D_real_mm
area_speck_mm2 = area_speck_px / px_per_mm**2

print(f"speck area  = {area_speck_px} px² "
      f"({area_speck_mm2:.2f} mm², if D={D_real_mm} mm)")
print(f"rectangle   = {coverage_rect:.1%} of its ROI")
print(f"whole sheet = {coverage_sheet:.3%} of the image")
