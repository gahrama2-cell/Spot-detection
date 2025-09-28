**Pixel counting & bilateral-filter tuning for speck detection**

*This repo contains three Jupyter notebooks:*

a) an interactive freehand polygon pixel counter to check areas directly on an image,

b) a bilateral filter parameter tuner that grid-searches preprocessing settings to match ground-truth (mm²),

c) a single-pass speck area report generator that produces crops, masks, and a CSV using a chosen denoising/filter method.

**Reading order**

Count_pixels_inside_a_freehand_polygon.ipynb — understand the interactive workflow and how pixel areas are obtained from masks.

Best_parameter_with_Bilateral Filter Tuning.ipynb — reproduce the automatic pipeline, tune parameters, and select good bilateral filter settings for 200 masked residues which pixel areas counted by Count_pixels_inside_a_freehand_polygon.ipynb.

Dark_Speck_Area_Report.ipynb — run a single sweep (with the chosen filter & parameters) to generate the final images and CSV report.

**1) Freehand polygon pixel counter**

*Notebook: Count_pixels_inside_a_freehand_polygon.ipynb*

*Goal: Draw a freehand region on an image and print the number of full-resolution pixels inside the drawn polygon.*

What it does:

a. Opens an image with OpenCV, auto-scales to your screen for display by keeping a copy internally.

b. While you hold left-click, it records the freehand path (in original-image coordinates).

c. On right-click, it closes the polygon, fills it as a mask, and prints the count of pixels inside.

d. Displays a blended overlay of the selected region.
   
**2) Bilateral filter parameter tuning**

*Notebook: Best_parameter_with_Bilateral Filter Tuning.ipynb*

*Goal: Find (d, sigmaColor, sigmaSpace) for OpenCV’s bilateral filter that best match my ground-truth speck areas (in mm²) over many region of interests.*

Assumptions & inputs:

Images and JSON annotations are in a folder (let's say named Area).

Each image (for example: sheet1.jpg) has a corresponding LabelMe-style JSON in Area whose file name starts with the image stem (sheet1*.json).

Rectangular region of interests are taken from ann["shapes"] where:

shape_type == "rectangle"

label starts with one of S, p, r, or 1

A CSV contains a row per region of interest with:

image — expected to match "{image_stem}_{label}_{rect_counter}.png"

dark_px_by hand_mm2 — ground-truth speck area (mm²) for that region of interest

Physical sheet diameter is D_MM = 164.4, giving area A_SHEET_MM2 = π * (D_MM/2)²

Each image’s pixel→mm² scale is computed as A_SHEET_MM2 / (H*W)

What it does:

a. Applies a bilateral filter with a given (d, sigmaColor, sigmaSpace) for each region of interest

b. Converts region of interest to grayscale and thresholds with Otsu → binary mask of dark speck

c. Counts speck pixels and converts to mm² using the per-image scale.

d. Writes the automatic result back into gt['dark_px_by auto_mm2'].

Saves:

Cropped region of interest: Area/cropped_rois/<base>.png. Visual mask overlay: Area/masked_specks/<base>_mask.png. It then grid-searches over:

d ∈ {5, 10, 15, 20, 25, 30}

sigmaColor ∈ {5, 10, …, 40}

sigmaSpace ∈ {5, 10, 15, 20, 25, 30}

For each combo, it computes per-region of interest relative error - abs(hand - auto) / hand * 100 and keeps the combo with the lowest STD of error acrossregion of intresets. Finally, it writes a CSV of the best run with columns: image, label, img_px, img_mm2, roi_px, roi_mm2, dark_px, dark_mm2

**3) Single-pass dark speck area report**

*Notebook: Dark_Speck_Area_Report.ipynb*

*Goal: Generate a one-shot report (images + CSV) using one chosen denoising/filter method and fixed parameters. This is handy after picking settings with the tuner, or if you want to compare different prefilters.*

Filter options-The notebook includes commented blocks for several prefilters:

1. cv2.bilateralFilter(...) (default in the template; e.g., d=10, sigmaColor=40, sigmaSpace=20)

2. cv2.GaussianBlur(...)

3. cv2.medianBlur(...)

4. cv2.fastNlMeansDenoising(...)

5. cv2.pyrMeanShiftFiltering(...)

Inputs & assumptions:

Images and JSON annotations live in Area/ (same expectations as the tuner).

Rectangular ROIs are read from LabelMe-style JSON (shape_type == "rectangle", label starts with S, p, r, or 1).

Physical sheet constants: D_MM = 164.4, A_SHEET_MM2 = π * (D_MM/2)².

What it does:

For each image:

a. Applies the chosen prefilter.

b. Converts ROI to grayscale and thresholds with Otsu to get a binary speck mask.

c. Counts speck pixels and converts to mm² using the scale above.

Saves:

Cropped ROI → Area/cropped_rois_<FilterName>/<base>.png

Mask overlay → Area/masked_specks_<FilterName>/<base>_mask.png

Appends a CSV row with:
image, label, img_px, img_mm2, roi_px, roi_mm2, dark_px, dark_mm2

Finally, it writes out_csv and prints a short summary of output paths.
