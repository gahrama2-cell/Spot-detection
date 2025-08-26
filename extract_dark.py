import cv2
import numpy as np
from pathlib import Path
import glob
import os



def extract(file):
    # ─── CONFIG (unchanged) ─────────────────────────────────────────────
    IMG_PATH   = Path(file)
    OUT_PNG    = Path(f"dark_cropped/{os.path.basename(file)}")        # transparent cut-out
    OUT_OVER   = Path(f"dark_cropped/overlay_{os.path.basename(file)}.png")         # colourful visualisation
    COLOR      = (0, 0, 255)    # BGR → red; change to (0,255,255)=yellow etc.
    ALPHA      = 0.5            # overlay opacity 0-1
    OTSU_ONLY  = True
    MANUAL_T   = 55
    # ────────────────────────────────────────────────────────────────────

    img  = cv2.imread(str(IMG_PATH))
    assert img is not None, f"{IMG_PATH} not found"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if OTSU_ONLY:
        _, mask = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, MANUAL_T, 255, cv2.THRESH_BINARY_INV)

    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 2)

    num, lbl, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(lbl == largest, 255, 0).astype("uint8")

    # --------------- 1. perfect cut-out (unchanged) --------------------
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    cv2.imwrite(str(OUT_PNG), rgba)

    # --------------- 2. colourful overlay ------------------------------
    overlay = img.copy()
    overlay[mask == 255] = COLOR                    # paint cluster solid colour
    viz = cv2.addWeighted(overlay, ALPHA, img, 1-ALPHA, 0)  # blend
    cv2.imwrite(str(OUT_OVER), viz)
    # -------------------------------------------------------------------

    print("✓ Saved:",
        f"\n   {OUT_PNG.name}   (transparent cluster)",
        f"\n   {OUT_OVER.name}  (colour overlay)")

if __name__ == "__main__":
	fls = glob.glob("datasets/Trained2/crops/*.jpg")
	for fl in fls:
		extract(fl)