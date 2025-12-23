#joining the cropped images csv with original images csv

import pandas as pd
from pathlib import Path


def merge_original_and_cropped_csv():
    """
    Final merge logic (FIXED):
    - Original CSV = source of truth
    - Cropped rows append immediately after parent image
    - Cropped rows inherit ALL metadata + clustering
    - Join is done via normalized image key
    """

    # === Inputs ===
    original_csv = Path(input("Enter ABS path to ORIGINAL meta_data_clustered.csv: ").strip())
    cropped_csv  = Path(input("Enter ABS path to CROPPED predictions CSV: ").strip())
    dest_dir     = Path(input("Enter ABS path to OUTPUT directory: ").strip())

    dest_dir.mkdir(parents=True, exist_ok=True)
    output_csv = dest_dir / f"{cropped_csv.stem}_merged.csv"

    # === Load CSVs ===
    df_orig = pd.read_csv(original_csv, dtype=str)
    df_crop = pd.read_csv(cropped_csv, dtype=str)

    # === Safety checks ===
    if "image" not in df_orig.columns:
        raise ValueError("Original CSV must contain 'image' column")

    if "parent_image" not in df_crop.columns:
        raise ValueError("Cropped CSV must contain 'parent_image' column")

    # === Normalize join keys ===
    df_orig["__key__"] = df_orig["image"].str.strip().str.lower()
    df_crop["__key__"] = df_crop["parent_image"].str.strip().str.lower()

    # === Metadata columns (copied from original) ===
    META_COLS = [
        "site_id",
        "camera_id",
        "subfolder_token",
        "capture_ts",
        "capture_date",
        "cluster_id"
    ]

    # === Index cropped rows by key ===
    crop_groups = df_crop.groupby("__key__")

    merged_rows = []
    used_crops = 0

    # === Merge logic ===
    for _, orig_row in df_orig.iterrows():
        parent_key = orig_row["__key__"]

        # 1️⃣ Keep original row untouched
        merged_rows.append(orig_row.drop("__key__").to_dict())

        # 2️⃣ Append cropped rows (if any)
        if parent_key in crop_groups.groups:
            crops = crop_groups.get_group(parent_key)

            for _, crop_row in crops.iterrows():
                merged = orig_row.to_dict()  # start from original

                # Overwrite prediction-level fields ONLY
                for col in crop_row.index:
                    if col in {
                        "filepath",
                        "image",
                        "no_of_detections",
                        "label_hierarchy",
                        "common_name",
                        "prediction_score"
                    }:
                        merged[col] = crop_row[col]

                # Explicitly enforce metadata inheritance
                for mcol in META_COLS:
                    merged[mcol] = orig_row.get(mcol)

                merged.pop("__key__", None)
                merged_rows.append(merged)
                used_crops += 1

    # === Save ===
    merged_df = pd.DataFrame(merged_rows)
    merged_df.to_csv(output_csv, index=False)

    print("\n✅ MERGE COMPLETE")
    print(f"Original rows : {len(df_orig)}")
    print(f"Cropped rows  : {len(df_crop)}")
    print(f"Used crops    : {used_crops}")
    print(f"Final rows    : {len(merged_df)}")
    print(f"Output file   : {output_csv}")


if __name__ == "__main__":
    merge_original_and_cropped_csv()

