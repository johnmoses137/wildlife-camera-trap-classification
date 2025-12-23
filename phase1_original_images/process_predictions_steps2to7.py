#####UPDATED WORKING SCRIPT WITH COMMENTS#######
#########################
####################################
##############################################
###################################
##########################

"""
================================================================================
SPECIESNET UNIFIED WORKFLOW (Steps 2-7)
================================================================================

PURPOSE:
--------
This script processes wildlife camera trap images through a complete pipeline:
    Step 2: Convert JSON predictions to CSV
    Step 3: Annotate images with bounding boxes and labels
    Step 4: Group annotated images by species
    Step 5: Create visualization charts
    Step 6-7: Extract metadata and cluster images by time

OUTPUT STRUCTURE:
-----------------
All outputs are saved under: <JSON_DIR>/<IMAGES_FOLDER_NAME>/
    ├── predictions.csv              (Step 2)
    ├── annotated_images/            (Step 3)
    ├── Grouped_Images/              (Step 4)
    ├── Visualizations/              (Step 5)
    └── meta_data_clustered.csv      (Step 6-7)

REQUIRED PACKAGES FROM speciesnet_env:
---------------------------------------
- os, json, time, pathlib: Standard library for file/path operations
- pandas: Data manipulation and CSV handling
- cv2 (opencv-python): Image processing and annotation
- matplotlib, squarify, numpy: Data visualization
- exifread: Extract EXIF metadata from images
- tqdm: Progress bars for long operations
- re: Regular expressions for text parsing
- shutil: File copy operations

================================================================================
"""

######W WORKING SCRIPT FOR NEW DATA SETS  ####################



# ===============================================================
# UNIFIED SPECIESNET WORKFLOW (Steps 2 → 7)
# Saves all outputs under: <JSON_DIR>/<IMAGES_FOLDER_NAME>/
# ===============================================================

# ===============================================================
# CORE IMPORTS
# ===============================================================
# Standard library imports
import os, json, time
from pathlib import Path
# Data manipulation
import pandas as pd


# ===============================================================
# STEP 2: JSON TO CSV CONVERSION
# ===============================================================
# PURPOSE: Converts SpeciesNet JSON predictions to structured CSV format
# 
# PACKAGES USED:
# - re: Regular expressions for parsing taxonomic hierarchies
# - json: Parse JSON prediction file from SpeciesNet model
# - pandas: Create and save DataFrame as CSV
# - time: Track execution time
# 
# INPUT:
# - pred_json_path: Path to classifier_output_topk.json from SpeciesNet
# - out_root: Directory where predictions.csv will be saved
# 
# OUTPUT:
# - predictions.csv with columns:
#     * filepath: Full path to original image
#     * image: Filename only
#     * no_of_detections: Count of animals detected in image
#     * categories: Comma-separated detection labels (e.g., "deer, raccoon")
#     * confidence_scores: Confidence for each detection (0-1)
#     * label_hierarchy: Full taxonomic hierarchy from SpeciesNet
#     * common_name: Extracted species common name
#     * prediction_score: Overall prediction confidence
# 
# HOW IT WORKS:
# 1. Loads JSON predictions from SpeciesNet model output
# 2. Extracts common name from taxonomic hierarchy string
# 3. Filters out checkpoint/temporary images
# 4. Creates structured rows for each prediction with all relevant data
# 5. Saves as CSV for downstream processing
# ===============================================================
def step2_json_to_csv(pred_json_path: Path, out_root: Path) -> Path:
    import re
    start = time.time()

    # Validate input file exists
    if not pred_json_path.exists():
        raise SystemExit(f"[error] File not found: {pred_json_path}")

    # Define output path
    csv_path = out_root / "predictions.csv"

    # ---------------------------------------------------------------
    # HELPER FUNCTION: Extract common name from label hierarchy
    # ---------------------------------------------------------------
    # SpeciesNet returns hierarchical labels like:
    # "animalia;chordata;mammalia;carnivora;felidae;lynx rufus"
    # This function extracts "Lynx Rufus" as the common name
    # 
    # LOGIC:
    # 1. Split hierarchy by semicolons
    # 2. Handle special case: "blank" images (no animal)
    # 3. Handle UUID identifiers (skip them)
    # 4. Return last meaningful taxonomic level
    # ---------------------------------------------------------------
    def extract_common_name(label_hierarchy: str) -> str:
        if not label_hierarchy:
            return ""
        # Split by semicolon and clean whitespace
        parts = [p.strip().lower() for p in label_hierarchy.split(";") if p.strip()]
        if not parts:
            return ""
        last = parts[-1]
        # Handle blank images
        if last == "blank":
            return "blank"
        # Skip UUID identifiers, look for actual species name
        if re.fullmatch(r"[a-f0-9\-]+", last):
            for item in reversed(parts):
                if item not in {"blank", "animal"} and re.search(r"[a-z]", item):
                    return item.title()
            return ""
        return last.title()

    # Load JSON predictions file
    with open(pred_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract predictions array (handle different JSON structures)
    preds = data.get("predictions", []) if isinstance(data, dict) else data
    if not preds:
        raise SystemExit(f"[error] No predictions found in {pred_json_path}")

    # ---------------------------------------------------------------
    # BUILD CSV ROWS
    # ---------------------------------------------------------------
    # Process each prediction and create a structured row
    # Each row represents one image with its detections and classification
    # ---------------------------------------------------------------
    rows = []
    for p in preds:
        # Extract image filename
        img_name = os.path.basename(p.get("filepath") or p.get("image") or "")

        # Skip checkpoint files (temporary files created during processing)
        if "checkpoint" in img_name.lower():
            continue

        # Get detections (bounding boxes) for this image
        dets = p.get("detections", []) or []
        # Get classification label hierarchy
        label_h = str(p.get("prediction", "")).strip()

        # Create row with all relevant data
        rows.append({
            "filepath": p.get("filepath") or p.get("image"),
            "image": img_name,
            "no_of_detections": len(dets),
            # Join all detection labels into comma-separated string
            "categories": ", ".join(str(d.get("label","")).strip() for d in dets),
            # Format confidence scores to 3 decimal places
            "confidence_scores": ", ".join(f"{float(d.get('conf',0) or 0):.3f}" for d in dets),
            "label_hierarchy": label_h,
            "common_name": extract_common_name(label_h),
            "prediction_score": p.get("prediction_score","")
        })

    # Convert to DataFrame and save as CSV
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f" STEP 2 – JSON→CSV in {time.time()-start:.2f}s → {csv_path}")
    return csv_path


# ===============================================================
# STEP 3: ANNOTATE IMAGES WITH BOUNDING BOXES
# ===============================================================
# PURPOSE: Draw bounding boxes and labels on original images
# 
# PACKAGES USED:
# - cv2 (opencv-python): Image reading, drawing rectangles/text, saving
# - json: Parse predictions file
# - pathlib: Path operations
# 
# INPUT:
# - images_path: Directory containing original camera trap images
# - pred_json_path: JSON file with predictions and bounding boxes
# - out_root: Directory where annotated_images/ folder will be created
# 
# OUTPUT:
# - annotated_images/ folder containing copies of images with:
#     * Red bounding boxes around detected animals
#     * Confidence scores above each box
#     * Species common name at top of image
# 
# HOW IT WORKS:
# 1. Load predictions JSON with bounding box coordinates
# 2. For each image:
#    a. Read image using OpenCV
#    b. Draw red rectangles for each detection (bbox format: [x, y, width, height] normalized 0-1)
#    c. Add white text showing confidence score
#    d. Add species label at top
#    e. Save annotated image
# 
# COORDINATE SYSTEM:
# - Bounding boxes are normalized (0-1) relative to image dimensions
# - bbox[0] = x position (left edge, fraction of width)
# - bbox[1] = y position (top edge, fraction of height)  
# - bbox[2] = width (fraction of total width)
# - bbox[3] = height (fraction of total height)
# ===============================================================
def step3_annotate_images(images_path: Path, pred_json_path: Path, out_root: Path) -> Path:
    import cv2
    start = time.time()

    # Load predictions JSON
    with open(pred_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data.get("predictions", [])
    if not preds:
        raise SystemExit("[error] No predictions in JSON")

    # Create output directory for annotated images
    annotated_dir = out_root / "annotated_images"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    # ---------------------------------------------------------------
    # PROCESS EACH IMAGE
    # ---------------------------------------------------------------
    for pred in preds:
        # Get filename and construct full path
        fn = Path(pred.get("filepath","") or pred.get("image","")).name
        img_path = images_path / fn
        if not img_path.exists():
            continue

        # Read image using OpenCV (BGR format)
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Get image dimensions for coordinate conversion
        h, w, _ = img.shape

        # ---------------------------------------------------------------
        # DRAW BOUNDING BOXES FOR EACH DETECTION
        # ---------------------------------------------------------------
        # Each detection has a bbox and confidence score
        # We draw a red rectangle and white text label
        # ---------------------------------------------------------------
        for det in pred.get("detections", []):
            bbox = det.get("bbox", [])
            conf = float(det.get("conf", 0) or 0)
            if len(bbox) == 4:
                # Convert normalized coordinates to pixel coordinates
                x1, y1 = int(bbox[0]*w), int(bbox[1]*h)
                x2, y2 = x1 + int(bbox[2]*w), y1 + int(bbox[3]*h)

                # Draw red rectangle (BGR: 0,0,255 = red)
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
                # Add confidence score as white text above box
                cv2.putText(img, f"{conf:.2f}", (x1, max(0, y1-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # ---------------------------------------------------------------
        # ADD SPECIES LABEL AT TOP OF IMAGE
        # ---------------------------------------------------------------
        # Extract common name from hierarchy (last part)
        # Display prominently at top-left
        # ---------------------------------------------------------------
        common = (pred.get("prediction","") or "").split(";")[-1]
        cv2.putText(img, f"{common}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # Save annotated image
        cv2.imwrite(str(annotated_dir / fn), img)
        count += 1

    print(f" STEP 3 – Annotated {count} images in {time.time()-start:.2f}s → {annotated_dir}")
    return annotated_dir


# ===============================================================
# STEP 4: GROUP ANNOTATED IMAGES BY SPECIES
# ===============================================================
# PURPOSE: Organize annotated images into species-specific folders
# 
# PACKAGES USED:
# - pandas: Read predictions CSV
# - shutil: Copy files to new locations
# - pathlib: Directory and file operations
# 
# INPUT:
# - csv_path: predictions.csv from Step 2
# - annotated_dir: Folder of annotated images from Step 3
# - out_root: Directory where Grouped_Images/ will be created
# 
# OUTPUT:
# - Grouped_Images/ folder with structure:
#     ├── Deer (45)/
#     │   ├── Deer_001_IMG0001.jpg
#     │   ├── Deer_002_IMG0002.jpg
#     │   └── ...
#     ├── Raccoon (23)/
#     │   ├── Raccoon_001_IMG0010.jpg
#     │   └── ...
#     └── Blank (12)/
#         └── ...
# 
# HOW IT WORKS:
# 1. Read predictions CSV to get image→species mapping
# 2. For each image:
#    a. Find its species from common_name column
#    b. Copy from annotated_images/ to Grouped_Images/[species]/
#    c. Rename with sequential numbering: [species]_001_[original].jpg
# 3. Rename folders to include count: "Deer" → "Deer (45)"
# 
# BENEFITS:
# - Easy visual review of detections by species
# - Quick quality control
# - Organized for manual verification
# ===============================================================
def step4_group_annotated(csv_path: Path, annotated_dir: Path, out_root: Path) -> Path:
    import shutil
    start = time.time()

    # Load predictions CSV
    df = pd.read_csv(csv_path)
    
    # Create main grouped output directory
    grouped = out_root / "Grouped_Images"
    grouped.mkdir(parents=True, exist_ok=True)

    # Track counts per species for sequential numbering
    counts = {}
    
    # ---------------------------------------------------------------
    # COPY AND RENAME IMAGES BY SPECIES
    # ---------------------------------------------------------------
    # Iterate through each row in predictions CSV
    # Copy corresponding annotated image to species folder
    # ---------------------------------------------------------------
    for _, r in df.iterrows():
        img = str(r.get("image", "")).strip()
        sp  = str(r.get("common_name", "")).strip()
        
        # Skip rows with missing data
        if not img or not sp:
            continue

        # Source: annotated image
        src = annotated_dir / img
        if not src.exists():
            continue

        # Destination: species subfolder
        dst_dir = grouped / sp
        dst_dir.mkdir(exist_ok=True)

        # Increment counter for this species
        counts[sp] = counts.get(sp, 0) + 1
        # Create new filename: [Species]_[###]_[original_name].jpg
        dst_img = dst_dir / f"{sp}_{counts[sp]:03d}_{img}"
        shutil.copy2(src, dst_img)

    # ---------------------------------------------------------------
    # RENAME FOLDERS TO INCLUDE COUNTS
    # ---------------------------------------------------------------
    # Change "Deer" → "Deer (45)" for easy identification
    # ---------------------------------------------------------------
    for sp, c in counts.items():
        src = grouped / sp
        dst = grouped / f"{sp} ({c})"
        if src.exists() and not dst.exists():
            src.rename(dst)

    print(f" STEP 4 – Grouped {sum(counts.values())} images "
          f"({len(counts)} species) in {time.time()-start:.2f}s → {grouped}")
    return grouped


# ===============================================================
# STEP 5: CREATE SPECIES DISTRIBUTION VISUALIZATIONS
# ===============================================================
# PURPOSE: Generate bar charts showing species abundance
# 
# PACKAGES USED:
# - matplotlib.pyplot: Create and save bar charts
# - squarify: (imported but not used - could be used for treemaps)
# - numpy: (imported for potential numeric operations)
# - re: Remove count numbers from folder names
# - pathlib: Directory operations
# 
# INPUT:
# - grouped_dir: Grouped_Images/ folder from Step 4
# - out_root: Directory where Visualizations/ will be created
# 
# OUTPUT:
# - Visualizations/ folder containing:
#     * bar_including_blanks.png: All species including blank images
#     * bar_excluding_blanks.png: Only species detections
# 
# HOW IT WORKS:
# 1. Count images in each species folder
# 2. Extract species name (remove count from folder name)
# 3. Create two datasets:
#    a. All species (including "blank" images with no animals)
#    b. Non-blank species only
# 4. Sort by count (descending)
# 5. Generate horizontal bar charts
# 6. Save as high-resolution PNGs (300 DPI)
# 
# VISUALIZATION DETAILS:
# - Horizontal bars for better species name readability
# - Inverted Y-axis (most abundant at top)
# - Large figure size (14x8) for clarity
# - High DPI for publication quality
# ===============================================================
def step5_visualize(grouped_dir, out_root):
    import re, matplotlib.pyplot as plt, squarify, numpy as np

    base = Path(grouped_dir)
    # Create output directory for visualizations
    viz_dir = out_root / "Visualizations"
    viz_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------------
    # COUNT IMAGES PER SPECIES
    # ---------------------------------------------------------------
    # Iterate through species folders and count JPG files
    # Remove "(count)" from folder names to get clean species names
    # ---------------------------------------------------------------
    species_counts = {}
    for f in base.iterdir():
        if f.is_dir():
            # Count both .jpg and .JPG extensions
            count = len(list(f.glob("*.jpg"))) + len(list(f.glob("*.JPG")))
            # Remove "(45)" from "Deer (45)" → "Deer"
            nm = re.sub(r"\(\d+\)", "", f.name).strip()
            species_counts[nm] = count

    # Exit if no species found
    if not species_counts:
        return

    # ---------------------------------------------------------------
    # PREPARE DATA FOR VISUALIZATIONS
    # ---------------------------------------------------------------
    # Create two datasets: with and without blank images
    # Sort by count (highest first)
    # ---------------------------------------------------------------
    # Dataset 1: Exclude blank images
    non_blank = {k: v for k, v in species_counts.items() if k.lower() != "blank"}
    # Dataset 2: Include everything
    sorted_all = dict(sorted(species_counts.items(), key=lambda x: x[1], reverse=True))
    sorted_nb  = dict(sorted(non_blank.items(), key=lambda x: x[1], reverse=True))

    # ---------------------------------------------------------------
    # HELPER FUNCTION: Create horizontal bar chart
    # ---------------------------------------------------------------
    def make_bar(data, title, fname):
        plt.figure(figsize=(14,8))
        # Create horizontal bars
        plt.barh(list(data.keys()), list(data.values()))
        # Invert Y-axis so highest count is at top
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.tight_layout()
        # Save at high resolution
        plt.savefig(viz_dir / fname, dpi=300)
        plt.close()

    # Generate both visualizations
    make_bar(sorted_all, "Species Distribution (Including Blanks)", "bar_including_blanks.png")
    make_bar(sorted_nb,  "Species Distribution (Excluding Blanks)", "bar_excluding_blanks.png")


# ===============================================================
# STEP 6-7: EXTRACT METADATA AND CLUSTER BY TIME
# ===============================================================
# PURPOSE: Combine predictions with image metadata and group by time
# 
# PACKAGES USED:
# - exifread: Extract EXIF metadata (camera settings, timestamps)
# - tqdm: Progress bar for metadata extraction
# - pandas: Data manipulation, merging, datetime operations
# 
# INPUT:
# - images_path: Directory containing original images
# - pred_csv: predictions.csv from Step 2
# - out_root: Directory where meta_data_clustered.csv will be saved
# 
# OUTPUT:
# - meta_data_clustered.csv containing:
#     * All prediction data (species, detections, confidence)
#     * EXIF metadata (DateTimeOriginal, camera settings, etc.)
#     * Derived fields:
#       - site_id: Site identifier from folder structure
#       - camera_id: Camera identifier from folder structure
#       - capture_ts: Unified timestamp (timezone-naive)
#       - capture_date: Date only
#       - cluster_id: Sequential ID for temporal groupings
#       - timestamp_source: Where timestamp came from (EXIF or filesystem)
# 
# HOW IT WORKS:
# 
# PART 1: SITE/CAMERA IDENTIFICATION
# ----------------------------------
# Extracts hierarchical identifiers from folder structure:
# /path/to/SITE_NAME/CAMERA_NAME/SUBFOLDER/images
#   └─── site_id      camera_id   subfolder_token
# 
# PART 2: EXIF METADATA EXTRACTION
# ---------------------------------
# For each image file:
#   1. Read EXIF tags using exifread library
#   2. Extract all available metadata (timestamps, camera model, etc.)
#   3. Fall back to filesystem modification time if EXIF missing
#   4. Store in DataFrame with one row per image
# 
# PART 3: DATA MERGING
# ---------------------
# Join predictions CSV with metadata DataFrame on:
#   - Normalized filename (lowercase, no path)
#   - site_id, camera_id, subfolder_token
# This creates a unified dataset with both predictions and metadata
# 
# PART 4: TIMESTAMP RESOLUTION (PRIORITY SYSTEM)
# -----------------------------------------------
# Tries multiple timestamp sources in priority order:
#   Priority 1: DateTimeOriginal (when photo was taken)
#   Priority 2: DateTimeDigitized (when photo was digitized)
#   Priority 3: DateTime (general timestamp)
#   Priority 4: FileModifyDate (filesystem timestamp)
# 
# Uses first available timestamp for each image.
# Records which source was used in 'timestamp_source' column.
# Forces timezone-naive format for consistency.
# 
# PART 5: TIME-BASED CLUSTERING
# ------------------------------
# Groups images into "sequences" based on time gaps:
#   1. Sort all images by site, camera, and timestamp
#   2. Calculate time gap between consecutive images
#   3. If gap > CLUSTER_MINUTES (default: 5), start new cluster
#   4. Assign sequential cluster_id to each group
# 
# USE CASE: Identifies when the same animal(s) triggered multiple
# photos in quick succession vs. separate wildlife events.
# 
# Example:
#   Image 1: 10:00:00 → cluster_id = 1
#   Image 2: 10:00:15 → cluster_id = 1 (gap: 15 sec < 5 min)
#   Image 3: 10:00:45 → cluster_id = 1 (gap: 30 sec < 5 min)
#   Image 4: 10:08:00 → cluster_id = 2 (gap: 7.25 min > 5 min)
# 
# CLUSTER_MINUTES can be adjusted based on animal behavior and
# camera trigger settings.
# ===============================================================
def step6_7_metadata_and_join(images_path: Path, pred_csv: Path, out_root: Path) -> Path:
    import exifread
    from tqdm import tqdm
    import pandas as pd

    # Time threshold for clustering (in minutes)
    CLUSTER_MINUTES = 5

    # ---------------------------------------------------------------
    # EXTRACT SITE AND CAMERA IDENTIFIERS FROM PATH
    # ---------------------------------------------------------------
    # Folder structure: .../SITE_ID/CAMERA_ID/SUBFOLDER/
    # Extract these identifiers to track camera trap locations
    # ---------------------------------------------------------------
    SITE_ID   = images_path.parents[1].name if len(images_path.parents) >= 2 else images_path.parents[0].name
    CAMERA_ID = images_path.parent.name
    SUBFOLDER = images_path.name

    # ---------------------------------------------------------------
    # HELPER: Normalize filenames for matching
    # ---------------------------------------------------------------
    # Converts "C:/path/to/IMG001.JPG" → "img001.jpg"
    # Ensures consistent matching between predictions and metadata
    # ---------------------------------------------------------------
    def normalize(s):
        return str(s).split("/")[-1].lower()

    # -------------------------------
    # EXIF extraction
    # -------------------------------
    # ---------------------------------------------------------------
    # HELPER: Extract EXIF metadata from a single image
    # ---------------------------------------------------------------
    # EXIF contains camera settings, timestamps, GPS, etc.
    # Common tags:
    #   - DateTimeOriginal: When photo was captured
    #   - DateTimeDigitized: When RAW was converted
    #   - DateTime: Generic timestamp
    #   - Make/Model: Camera brand and model
    #   - ISO, Aperture, ShutterSpeed: Camera settings
    #   - GPSLatitude/Longitude: Location (if available)
    # 
    # This function:
    #   1. Opens image file in binary mode
    #   2. Processes EXIF tags using exifread
    #   3. Stores all tags in dictionary (key = tag name, value = value)
    #   4. Falls back to filesystem timestamp if EXIF missing
    # ---------------------------------------------------------------
    def extract_exif(p):
        d = {"FileName": p.name}
        try:
            with open(p, "rb") as f:
                # Process EXIF tags (details=False for speed)
                tags = exifread.process_file(f, details=False)
            # Store all EXIF tags, removing prefix (e.g., "EXIF DateTimeOriginal" → "DateTimeOriginal")
            for k, v in tags.items():
                d[k.split()[-1]] = str(v)
        except Exception:
            pass

        # Filesystem fallback (timezone-naive)
        # If EXIF is missing, use file modification time
        try:
            d["FileModifyDate"] = pd.to_datetime(
                p.stat().st_mtime, unit="s"
            )
        except Exception:
            d["FileModifyDate"] = pd.NaT

        return d

    # -------------------------------
    # Build metadata table
    # -------------------------------
    # ---------------------------------------------------------------
    # EXTRACT METADATA FROM ALL IMAGES
    # ---------------------------------------------------------------
    # Process every image file in the images directory
    # tqdm provides progress bar for large datasets
    # Creates DataFrame with one row per image
    # ---------------------------------------------------------------
    meta = pd.DataFrame([extract_exif(p) for p in tqdm(images_path.glob("*"))])
    # Add normalized filename for joining
    meta["__key__"] = meta["FileName"].map(normalize)
    # Add site/camera identifiers
    meta["site_id"] = SITE_ID
    meta["camera_id"] = CAMERA_ID
    meta["subfolder_token"] = SUBFOLDER

    # -------------------------------
    # Load predictions CSV
    # -------------------------------
    # ---------------------------------------------------------------
    # LOAD PREDICTIONS AND PREPARE FOR JOINING
    # ---------------------------------------------------------------
    # Read predictions CSV (all columns as strings initially)
    # Add same identifiers for joining with metadata
    # ---------------------------------------------------------------
    pred = pd.read_csv(pred_csv, dtype=str)
    pred["__key__"] = pred["image"].map(normalize)
    pred["site_id"] = SITE_ID
    pred["camera_id"] = CAMERA_ID
    pred["subfolder_token"] = SUBFOLDER

    # -------------------------------
    # Join predictions + metadata
    # -------------------------------
    # ---------------------------------------------------------------
    # MERGE PREDICTIONS WITH METADATA
    # ---------------------------------------------------------------
    # Left join: keep all predictions, add metadata where available
    # Join on: normalized filename + site/camera identifiers
    # Result: unified dataset with both predictions and EXIF data
    # ---------------------------------------------------------------
    joined = pred.merge(
        meta,
        on=["__key__", "site_id", "camera_id", "subfolder_token"],
        how="left"
    )

    # -------------------------------
    # ROBUST TIMESTAMP RESOLUTION
    # -------------------------------
    # ---------------------------------------------------------------
    # RESOLVE CAPTURE TIMESTAMP FROM MULTIPLE SOURCES
    # ---------------------------------------------------------------
    # Camera trap images may have timestamps in different EXIF fields
    # or only in filesystem metadata. This section implements a
    # priority system to find the best available timestamp.
    # 
    # PRIORITY ORDER:
    # 1. DateTimeOriginal - Most reliable, actual capture time
    # 2. DateTimeDigitized - When image was processed
    # 3. DateTime - Generic timestamp field
    # 4. FileModifyDate - Filesystem timestamp (least reliable)
    # 
    # For each image, uses first available timestamp and records
    # which source was used for transparency and quality control.
    # ---------------------------------------------------------------
    joined["capture_ts"] = pd.NaT  # Initialize as Not-a-Time
    joined["timestamp_source"] = None

    # Priority 1: EXIF timestamps (most reliable)
    # Try multiple EXIF timestamp fields in order of preference
    for col in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
        if col in joined.columns:
            # Parse timestamp, coerce errors to NaT
            ts = pd.to_datetime(joined[col], errors="coerce")
            # Only fill missing timestamps (don't overwrite existing)
            mask = joined["capture_ts"].isna() & ts.notna()
            joined.loc[mask, "capture_ts"] = ts[mask]
            # Record which field provided the timestamp
            joined.loc[mask, "timestamp_source"] = col

    # Priority 2: filesystem timestamp (fallback)
    # Used when EXIF timestamps are completely missing
    if "FileModifyDate" in joined.columns:
        ts = pd.to_datetime(joined["FileModifyDate"], errors="coerce")
        mask = joined["capture_ts"].isna() & ts.notna()
        joined.loc[mask, "capture_ts"] = ts[mask]
        joined.loc[mask, "timestamp_source"] = "FileModifyDate"

    # Validation: ensure at least some timestamps were found
    if joined["capture_ts"].isna().all():
        raise ValueError("No valid datetime metadata (EXIF or filesystem) found")

    # ✅ FORCE timezone-naive
    # Remove any timezone information for consistency
    # (Camera traps typically don't record timezone)
    joined["capture_ts"] = pd.to_datetime(joined["capture_ts"], errors="coerce")

    # Extract date-only field for daily summaries
    joined["capture_date"] = joined["capture_ts"].dt.date

    # -------------------------------
    # TIME-BASED CLUSTERING
    # -------------------------------
    # ---------------------------------------------------------------
    # GROUP IMAGES INTO TEMPORAL CLUSTERS
    # ---------------------------------------------------------------
    # When an animal triggers a camera trap, it often takes multiple
    # photos in quick succession (burst mode). These should be grouped
    # together as a single "detection event" or "sequence".
    # 
    # ALGORITHM:
    # 1. Sort images by site, camera, and timestamp
    # 2. Calculate time gap between each consecutive image
    # 3. When gap > CLUSTER_MINUTES, increment cluster_id
    # 4. Images within CLUSTER_MINUTES belong to same cluster
    # 
    # EXAMPLE with CLUSTER_MINUTES = 5:
    #   Time      Gap      Cluster
    #   10:00:00  -        1
    #   10:00:15  15 sec   1  (same animal)
    #   10:02:30  2.25 min 1  (still same animal)
    #   10:08:00  5.5 min  2  (new animal/event)
    #   10:09:00  1 min    2  (same animal)
    # 
    # PARAMETERS:
    # - CLUSTER_MINUTES: Adjustable based on camera settings
    #   * Fast trigger cameras: use 3-5 minutes
    #   * Slow cameras: use 10-15 minutes
    #   * Depends on expected animal behavior
    # ---------------------------------------------------------------
    
    # Sort by location and time
    joined = joined.sort_values(["site_id", "camera_id", "capture_ts"])

    # Calculate time gap (in minutes) between consecutive images
    # Group by site/camera to avoid gaps between different cameras
    gap = (
        joined
        .groupby(["site_id", "camera_id"])["capture_ts"]
        .diff()  # Time difference from previous image
        .dt.total_seconds()  # Convert to seconds
        .div(60)  # Convert to minutes
    )

    # Create cluster ID: increment when gap > threshold or at start of group
    # cumsum creates sequential IDs: [True, False, False, True, False] → [1, 1, 1, 2, 2]
    joined["cluster_id"] = (gap.isna() | (gap > CLUSTER_MINUTES)).cumsum()

    # -------------------------------
    # Save output
    # -------------------------------
    # ---------------------------------------------------------------
    # SAVE FINAL MERGED DATASET
    # ---------------------------------------------------------------
    # Output CSV contains:
    #   - All prediction data (species, detections, confidence)
    #   - All EXIF metadata (camera settings, timestamps)
    #   - Derived fields (site_id, camera_id, cluster_id)
    #   - Resolved timestamps with source tracking
    # 
    # This is the final, analysis-ready dataset for:
    #   - Occupancy modeling
    #   - Activity pattern analysis
    #   - Species co-occurrence studies
    #   - Camera trap effectiveness evaluation
    # ---------------------------------------------------------------
    out_csv = out_root / "meta_data_clustered.csv"
    joined.to_csv(out_csv, index=False)

    print(f" STEP 6–7 → {out_csv}")
    return out_csv

          

# ===============================================================
# MAIN WORKFLOW ORCHESTRATOR
# ===============================================================
# PURPOSE: Execute all steps in sequence with user inputs
# 
# HOW IT WORKS:
# 1. Prompt user for input paths
# 2. Create output directory structure
# 3. Execute steps 2-7 in order, passing outputs between steps
# 4. Each step builds on previous step's output
# 
# OUTPUT DIRECTORY STRUCTURE:
# ---------------------------
# <JSON_DIR>/<IMAGES_FOLDER_NAME>/
#   ├── predictions.csv              (Step 2)
#   ├── annotated_images/            (Step 3)
#   │   ├── IMG001.jpg
#   │   ├── IMG002.jpg
#   │   └── ...
#   ├── Grouped_Images/              (Step 4)
#   │   ├── Deer (45)/
#   │   ├── Raccoon (23)/
#   │   └── ...
#   ├── Visualizations/              (Step 5)
#   │   ├── bar_including_blanks.png
#   │   └── bar_excluding_blanks.png
#   └── meta_data_clustered.csv      (Step 6-7)
# 
# WORKFLOW DEPENDENCIES:
# ----------------------
# Step 2 → Step 3 (provides JSON for annotation)
# Step 2 → Step 4 (provides CSV for grouping)
# Step 3 → Step 4 (provides annotated images)
# Step 4 → Step 5 (provides grouped folders for counting)
# Step 2 → Step 6-7 (provides predictions for merging)
# ===============================================================
def run_workflow():
    # ---------------------------------------------------------------
    # USER INPUT COLLECTION
    # ---------------------------------------------------------------
    # Get absolute paths to required inputs:
    # 1. images_path: Original camera trap images
    # 2. pred_json_path: SpeciesNet model output JSON
    # ---------------------------------------------------------------
    images_path = Path(input("Enter ABSOLUTE path to input images folder: ").strip())
    pred_json_path = Path(input("Enter ABSOLUTE path to predictions.json: ").strip())

    # ---------------------------------------------------------------
    # OUTPUT DIRECTORY SETUP
    # ---------------------------------------------------------------
    # Create output folder next to JSON file
    # Name = images folder name (spaces replaced with underscores)
    # Example: "Camera A Deployment 1" → "Camera_A_Deployment_1"
    # ---------------------------------------------------------------
    out_root = pred_json_path.parent / images_path.name.replace(" ", "_")
    out_root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # EXECUTE WORKFLOW STEPS
    # ---------------------------------------------------------------
    # Each step returns a path that may be used by subsequent steps
    # Progress is printed to console after each step completes
    # ---------------------------------------------------------------
    
    # Step 2: Convert JSON predictions to CSV
    csv_path = step2_json_to_csv(pred_json_path, out_root)
    
    # Step 3: Draw bounding boxes on images
    ann_dir = step3_annotate_images(images_path, pred_json_path, out_root)
    
    # Step 4: Group annotated images by species
    grp_dir = step4_group_annotated(csv_path, ann_dir, out_root)
    
    # Step 5: Create species distribution charts
    step5_visualize(grp_dir, out_root)
    
    # Step 6-7: Extract metadata and cluster by time
    step6_7_metadata_and_join(images_path, csv_path, out_root)

    print("\n✓ WORKFLOW COMPLETE")


# ===============================================================
# SCRIPT ENTRY POINT
# ===============================================================
# Standard Python idiom: only run workflow when script is executed
# directly (not when imported as a module)
# ===============================================================
if __name__ == "__main__":
    run_workflow()
