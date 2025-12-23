"""
Crop Detections (Exact Bounding Boxes, Ignore Detection 0)
----------------------------------------------------------
Reads SpeciesNet-style predictions.json, where each detection bbox
is formatted as [x_min, y_min, width, height] (normalized).
Crops all detections starting from index 1 (skips detection 0) and saves
each camera's crops under:
    <destination>/<camera_name>/cropped_images/
Example run:
    Enter ABSOLUTE path to predictions.json: /mnt/home/bollaraj/trail/predictions_output.json
    Enter ABSOLUTE path to destination root: /mnt/home/bollaraj/trail/output_crops/
"""
import os
import json
import cv2
from pathlib import Path

def crop_detections_exact_bboxes():
    # === User Inputs ===
    json_path = Path(input("Enter ABSOLUTE path to predictions.json: ").strip())
    dest_root = Path(input("Enter ABSOLUTE path to destination root: ").strip())

    if not json_path.exists():
        raise FileNotFoundError(f"Predictions JSON not found: {json_path}")

    dest_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Cropped images will be saved under: {dest_root}")

    # === Load JSON ===
    with open(json_path, "r") as f:
        data = json.load(f)

    predictions = data.get("predictions", data)
    total_images = 0
    total_crops = 0
    processed_cameras = set()

    # === Iterate through predictions ===
    for _, item in enumerate(predictions.values() if isinstance(predictions, dict) else predictions, start=1):
        detections = item.get("detections", [])
        filepath = item.get("filepath")

        if not detections or not filepath or not os.path.exists(filepath):
            continue

        image = cv2.imread(filepath)
        if image is None:
            continue

        total_images += 1
        h_img, w_img = image.shape[:2]
        image_name = Path(filepath).name

        # === Extract camera folder name ===
        parts = Path(filepath).parts
        cam_folder_name = "Unknown_Camera"
        for idx in range(len(parts) - 1):
            if parts[idx].lower().startswith("lma"):
                lma = parts[idx].replace(" ", "")
                if idx + 1 < len(parts):
                    cam = parts[idx + 1].replace(" ", "")
                    cam_folder_name = f"{lma}_{cam}"
                    break

        # === Define destination paths ===
        cam_root = dest_root / cam_folder_name
        cropped_dir = cam_root / "cropped_images"
        cropped_dir.mkdir(parents=True, exist_ok=True)
        processed_cameras.add(cam_folder_name)

        # === Crop detections starting from index 1 (skip 0) ===
        for i, det in enumerate(detections):
            if i == 0:
                continue

            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            # bbox = [x_min, y_min, width, height]  (normalized)
            x1 = int(bbox[0] * w_img)
            y1 = int(bbox[1] * h_img)
            x2 = int((bbox[0] + bbox[2]) * w_img)
            y2 = int((bbox[1] + bbox[3]) * h_img)

            # Clip to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_filename = f"{Path(image_name).stem}-det{i}.jpg"
            cv2.imwrite(str(cropped_dir / crop_filename), crop)
            total_crops += 1

    # === Summary ===
    print("\n========= Processing Summary =========")
    print(f"Predictions JSON     : {json_path}")
    print(f"Images processed     : {total_images}")
    print(f"Total crops saved    : {total_crops}")
    print(f"Cropped output root  : {dest_root}")
    if processed_cameras:
        print("Processed Cameras    : " + ", ".join(sorted(processed_cameras)))
    print("======================================\n")

if __name__ == "__main__":
    crop_detections_exact_bboxes()
