
#### UPDATED SCRIPT ####


# ===============================================================
# UNIFIED SPECIESNET WORKFLOW (CROPPED IMAGES)
# Steps 2 → 7 (RESTORED)
# - Steps 2–5 reused from original-images workflow
# - Steps 6–7 inherited from original images (UNCHANGED)
# ===============================================================

import os, json, time, shutil
from pathlib import Path
import pandas as pd


# ---------- STEP 2: JSON → CSV ----------
def step2_json_to_csv(pred_json_path: Path, out_root: Path) -> Path:
    import re
    start = time.time()
    csv_path = out_root / "predictions.csv"

    def extract_common_name(label_hierarchy: str) -> str:
        if not label_hierarchy:
            return ""
        parts = [p.strip().lower() for p in label_hierarchy.split(";") if p.strip()]
        last = parts[-1]
        return "blank" if last == "blank" else last.title()

    with open(pred_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data.get("predictions", data)
    rows, skipped = [], 0

    for p in preds:
        img_name = os.path.basename(p.get("filepath") or p.get("image") or "")
        if "checkpoint" in img_name.lower():
            skipped += 1
            continue

        dets = p.get("detections", []) or []
        rows.append({
            "filepath": p.get("filepath") or p.get("image"),
            "image": img_name,
            "no_of_detections": len(dets),
            "label_hierarchy": p.get("prediction",""),
            "common_name": extract_common_name(p.get("prediction","")),
            "prediction_score": p.get("prediction_score","")
        })

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f" STEP 2 – JSON→CSV ({len(rows)} rows, {skipped} checkpoints skipped)")
    return csv_path


# ---------- STEP 3: ANNOTATE CROPPED IMAGES ----------
def step3_annotate_images(images_path: Path, pred_json_path: Path, out_root: Path) -> Path:
    import cv2
    start = time.time()

    with open(pred_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds = data.get("predictions", [])
    annotated_dir = out_root / "annotated_images"
    annotated_dir.mkdir(exist_ok=True)

    count = 0
    for pred in preds:
        fn = Path(pred.get("filepath") or pred.get("image") or "").name
        img_path = images_path / fn
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w, _ = img.shape

        for det in pred.get("detections", []):
            bbox = det.get("bbox", [])
            conf = float(det.get("conf", 0) or 0)
            if len(bbox) == 4:
                x1, y1 = int(bbox[0]*w), int(bbox[1]*h)
                x2, y2 = x1 + int(bbox[2]*w), y1 + int(bbox[3]*h)
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(img, f"{conf:.2f}", (x1, max(0,y1-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        label = (pred.get("prediction","") or "").split(";")[-1]
        cv2.putText(img, label, (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imwrite(str(annotated_dir / fn), img)
        count += 1

    print(f" STEP 3 – Annotated {count} cropped images")
    return annotated_dir


# ---------- STEP 4: GROUP ANNOTATED ----------
def step4_group_annotated(csv_path: Path, annotated_dir: Path, out_root: Path) -> Path:
    df = pd.read_csv(csv_path)
    grouped = out_root / "Grouped_Images"
    grouped.mkdir(exist_ok=True)

    counts = {}
    for _, r in df.iterrows():
        img = str(r.get("image","")).strip()
        sp  = str(r.get("common_name","")).strip()
        if not img or not sp:
            continue

        src = annotated_dir / img
        if not src.exists():
            continue

        dst = grouped / sp
        dst.mkdir(exist_ok=True)

        counts[sp] = counts.get(sp, 0) + 1
        shutil.copy2(src, dst / f"{sp}_{counts[sp]:03d}_{img}")

    for sp, c in counts.items():
        d = grouped / sp
        d.rename(grouped / f"{sp} ({c})")

    print(f" STEP 4 – Grouped {sum(counts.values())} cropped images")
    return grouped


# ---------- STEP 5: VISUALIZE ----------
def step5_visualize(grouped_dir: Path, out_root: Path):
    import matplotlib.pyplot as plt, re

    viz = out_root / "Visualizations"
    viz.mkdir(exist_ok=True)

    counts = {}
    for d in grouped_dir.iterdir():
        if d.is_dir():
            nm = re.sub(r"\(\d+\)", "", d.name).strip()
            counts[nm] = len(list(d.glob("*.jpg")))

    if not counts:
        return

    plt.figure(figsize=(12,6))
    plt.barh(list(counts.keys()), list(counts.values()))
    plt.gca().invert_yaxis()
    plt.title("Cropped Species Distribution")
    plt.tight_layout()
    plt.savefig(viz / "cropped_species_distribution.png", dpi=300)
    plt.close()


# ---------- STEP 6–7: INHERIT METADATA (UNCHANGED) ----------
def step6_7_inherit_metadata(cropped_csv, original_meta_csv, out_root):
    cropped = pd.read_csv(cropped_csv, dtype=str)
    original = pd.read_csv(original_meta_csv, dtype=str)

    cropped["parent_image"] = cropped["image"].str.replace(r"-det\d+", "", regex=True)
    cropped["parent_key"] = cropped["parent_image"].str.lower()
    original["key"] = original["image"].str.lower()

    cols = ["site_id","camera_id","subfolder_token","capture_ts","capture_date","cluster_id"]

    joined = cropped.merge(
        original[["key"] + cols],
        left_on="parent_key",
        right_on="key",
        how="left"
    ).drop(columns=["key","parent_key"])

    out = out_root / "cropped_predictions_with_metadata.csv"
    joined.to_csv(out, index=False)
    print(f" STEP 6–7 – Metadata inherited → {out}")
    return out


# ---------- RUN ----------
def run_workflow():
    cropped_imgs = Path(input("Enter ABS path to CROPPED images folder: ").strip())
    pred_json = Path(input("Enter ABS path to CROPPED predictions.json: ").strip())
    original_meta = Path(input("Enter ABS path to ORIGINAL meta_data_clustered.csv: ").strip())

    out_root = pred_json.parent / "cropped_outputs"
    out_root.mkdir(exist_ok=True)

    csv = step2_json_to_csv(pred_json, out_root)
    ann = step3_annotate_images(cropped_imgs, pred_json, out_root)
    grp = step4_group_annotated(csv, ann, out_root)
    step5_visualize(grp, out_root)
    step6_7_inherit_metadata(csv, original_meta, out_root)

    print("\n✓ CROPPED WORKFLOW COMPLETE")


if __name__ == "__main__":
    run_workflow()
