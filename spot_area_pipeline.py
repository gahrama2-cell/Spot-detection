#!/usr/bin/env python3
"""
spot_area_pipeline.py
=====================
• Part A  (images)  : width_px, height_px, circle_area_px2   → images.csv
• Part B  (rectangles): pixel_area → mm² (raw & adjusted)    → rects.csv
   – Rectangle JSON must have an "imagePath" that points to the file we just scanned.

Run:
    python spot_area_pipeline.py \
           --images   ./imgs \
           --jsons    ./annotations \
           --out-img  images.csv \
           --out-rect rects.csv
Dependencies: Pillow (pip install pillow)
"""
import argparse, csv, json, math, sys
from pathlib import Path
from typing   import Dict, List, Tuple
from PIL      import Image               # Pillow

# -----------------  CONSTANTS  -----------------
RADIUS_MM      = 82.5                       # 16.5 cm disc
CIRCLE_AREA_MM = math.pi * RADIUS_MM**2     

FILLING_FACTOR = 0.70                       # tweak if needed
THRESHOLD_MM2  = 0.40
# ------------------------------------------------

def scan_images(img_dir: Path) -> Dict[str, Tuple[int, int, float]]:
    """
    Returns {filename : (w, h, circle_pixel_area)} for *all* rasters in img_dir.
    Circle pixel area is π·(min(w,h)/2)².
    """
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif"}
    records = {}
    for p in img_dir.rglob("*"):
        if p.suffix.lower() not in exts:
            continue
        with Image.open(p) as im:
            w, h = im.size
        r_px  = min(w, h) / 2.0
        circA = math.pi * r_px**2
        records[p.name] = (w, h, circA)      # store by *basename* for easy lookup
    return records

def write_images_csv(records: Dict[str, Tuple[int,int,float]], csv_path: Path) -> None:
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "width_px", "height_px", "circle_area_px2"])
        for fname, (w_px, h_px, area_px2) in sorted(records.items()):
            w.writerow([fname, w_px, h_px, f"{area_px2:.1f}"])
    print(f"✓ images.csv → {csv_path.resolve()}  ({len(records)} lines)")

# ---------- rectangles ------------------------------------------------------
def load_rectangles(json_path: Path) -> List[Tuple[str,float]]:
    with json_path.open() as f:
        data = json.load(f)
    entries = data.get("shapes", data)
    rects = []
    for item in entries:
        if item.get("shape_type") != "rectangle":
            continue
        label = str(item.get("label", "rect"))
        (x1, y1), (x2, y2) = item["points"]
        rects.append((label, abs(x2-x1)*abs(y2-y1)))
    return rects, Path(data.get("imagePath", "")).name  # second value = image file

def process_rectangles(json_dir: Path,
                       img_meta: Dict[str, Tuple[int,int,float]],
                       out_csv: Path) -> None:
    total_adj, big_n, big_sum, rect_counter = 0, 0, 0, 0

    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["json_file", "label", "pixel_area",
                    "mm2_raw", "mm2_adj"])

        for jp in json_dir.rglob("*.json"):
            rects, img_name = load_rectangles(jp)
            if img_name not in img_meta:
                print(f"⚠ {jp.name}: image '{img_name}' not found, skipped", file=sys.stderr)
                continue

            circ_px_area = img_meta[img_name][2]
            mm2_per_px   = CIRCLE_AREA_MM / circ_px_area

            for lab, px_area in rects:
                mm2_raw = px_area * mm2_per_px
                mm2_adj = mm2_raw * FILLING_FACTOR
                w.writerow([jp.name, lab,
                            f"{px_area:.1f}", f"{mm2_raw:.3f}", f"{mm2_adj:.3f}"])

                total_adj += mm2_adj
                rect_counter += 1
                if mm2_adj >= THRESHOLD_MM2:
                    big_n   += 1
                    big_sum += mm2_adj

    print(f"✓ rects.csv  → {out_csv.resolve()}  ({rect_counter} rectangles)")
    print(f"   Total adjusted area: {total_adj:.3f} mm²")
    print(f"   {big_n} spots ≥ {THRESHOLD_MM2} mm² "
          f"(sum {big_sum:.3f} mm²)")

# ---------------------------------------------------------------------------
def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--images",   type=Path, required=True,
                    help="Folder containing the raster images")
    pa.add_argument("--jsons",    type=Path, required=True,
                    help="Folder containing LabelMe JSONs")
    pa.add_argument("--out-img",  type=Path, default=Path("images.csv"))
    pa.add_argument("--out-rect", type=Path, default=Path("rects.csv"))
    args = pa.parse_args()

    img_meta = scan_images(args.images)
    if not img_meta:
        sys.exit("No images found – check the --images path/extension list")

    write_images_csv(img_meta, args.out_img)
    process_rectangles(args.jsons, img_meta, args.out_rect)

if __name__ == "__main__":
    main()
