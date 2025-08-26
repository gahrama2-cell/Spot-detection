import cv2
import os
import random
import json
import numpy as np
from tqdm  import tqdm





def is_black_region(region, threshold=15, black_ratio_threshold=0.7):
    """
    Returns True if the region is mostly black.
    """
    black_pixels = np.all(region <= threshold, axis=2)
    black_ratio = np.sum(black_pixels) / (region.shape[0] * region.shape[1])
    return black_ratio > black_ratio_threshold

if __name__=='__main__':
    # === CONFIGURATION ===
    cropped_dir = 'datasets/Trained2/crops'
    base_image_path = 'Test Pics/WhatsApp Image 2025-02-19 at 11.12.14 AM (1).png'
    coco_json_path = 'datasets/Trained2/annotations.json'
    updated_json_path = 'datasets/Trained2/updated_annotations.json'
    max_crops = random.randint(7, 150)

    # === LOAD BASE IMAGE ===
    base = cv2.imread(base_image_path)
    if base is None:
        raise FileNotFoundError(f"Base image not found: {base_image_path}")
    base_h, base_w = base.shape[:2]

    # === LOAD COCO JSON ===
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    # === Setup IDs ===
    next_image_id = max([img['id'] for img in coco['images']]) + 1
    next_ann_id = max([ann['id'] for ann in coco['annotations']]) + 1

    # === Add New Image Entry ===


    # === SELECT CROPPED IMAGES ===
    cropped_files = [f for f in os.listdir(cropped_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(cropped_files)
    num_to_blend = random.randint(7, 150)
    selected_crops = cropped_files[:num_to_blend]

    for st in tqdm(range(750)):
        base = cv2.imread(base_image_path)
        random.shuffle(cropped_files)
        num_to_blend = random.randint(7, 150)
        selected_crops = cropped_files[:num_to_blend]
        output_image_name = f'blended_0{st}.jpg'
        output_image_path = f'datasets/Trained2/images/{output_image_name}'
        coco['images'].append({
        'id': next_image_id,
        'file_name': f"images/{output_image_name}",
        'height': base_h,
        'width': base_w
        })
        # === PASTE CROPS & ADD ANNOTATIONS ===
        for crop_file in selected_crops:
            # Parse category ID from filename
            cat_id = 0  # fallback if no category in filename
            crop = cv2.imread(os.path.join(cropped_dir, crop_file))
            if crop is None:
                continue

            # Resize
            # scale = random.uniform(0.5, 1.5)
            # crop_resized = cv2.resize(crop, (0, 0), fx=scale, fy=scale)
            ch, cw = crop.shape[:2]

            # Random position
            max_x = base_w - cw
            max_y = base_h - ch
            mask = 255 * np.ones((ch, cw), dtype=np.uint8)
            if max_x <= 0 or max_y <= 0:
                continue
            x_offset = random.randint(0, max_x)
            y_offset = random.randint(0, max_y)
            while is_black_region(base[y_offset:y_offset+ch, x_offset:x_offset+cw]):
                 x_offset = random.randint(0, max_x)
                 y_offset = random.randint(0, max_y)
            # Paste
            roi = base[y_offset:y_offset+ch, x_offset:x_offset+cw]
            blended = cv2.addWeighted(roi, 0.6, crop, 0.4, 0)
            center = (x_offset + cw // 2, y_offset + ch // 2)
            # base[y_offset:y_offset+ch, x_offset:x_offset+cw] = blended
            base = cv2.seamlessClone(crop, base, mask, center, cv2.NORMAL_CLONE)
            # Add annotation
            coco['annotations'].append({
                'id': next_ann_id,
                'image_id': next_image_id,
                'category_id': 0,
                'bbox': [x_offset, y_offset, cw, ch],
                'area': cw * ch,
            })
            next_ann_id += 1
        next_image_id+=1

        # === SAVE BLENDED IMAGE & UPDATED JSON ===
        cv2.imwrite(output_image_path, base)
    with open(updated_json_path, 'w') as f:
        json.dump(coco, f)

    print(f"Saved blended image to {output_image_path}")
    print(f"Updated annotations saved to {updated_json_path}")
