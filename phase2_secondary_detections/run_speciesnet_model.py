"""
Run SpeciesNet model on a given image folder
--------------------------------------------
This script runs the SpeciesNet model on the provided image folder
and saves the predictions JSON directly into the destination directory
without creating a subfolder.
The JSON filename includes the image folder name and '_predictions.json'.
"""
import subprocess
import sys
from pathlib import Path

def run_speciesnet():
    # === Inputs ===
    img_in = input("Enter ABSOLUTE image folder path: ").strip()
    dest_in = input("Enter ABSOLUTE destination root path: ").strip()

    image_folder = Path(img_in).expanduser().resolve()
    dest_root = Path(dest_in).expanduser().resolve()
    country = "USA"

    # === Validate Inputs ===
    if not image_folder.exists():
        raise SystemExit(f"Error: Image folder not found: {image_folder}")
    dest_root.mkdir(parents=True, exist_ok=True)

    # === Construct Prediction File Path ===
    # Example: LMA25_cam69_croppedimages_predictions.json
    json_filename = f"{image_folder.name}_predictions.json"
    pred_json = dest_root / json_filename

    # === Locate SpeciesNet Environment ===
    speciesnet_py = Path.home() / "speciesnet_env/bin/python"
    if not speciesnet_py.exists():
        raise SystemExit(f"Error: Could not find SpeciesNet environment: {speciesnet_py}")

    # === Command ===
    cmd = [
        str(speciesnet_py),
        "-m", "speciesnet.scripts.run_model",
        "--folders", str(image_folder),
        "--predictions_json", str(pred_json),
        "--country", country,
        "--bypass_prompts",
        "--progress_bars"
    ]

    # === Run the Model ===
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"SpeciesNet failed (rc={proc.returncode})")

    # === Validate Output ===
    if not pred_json.exists() or pred_json.stat().st_size == 0:
        raise SystemExit(f"Error: predictions.json missing or empty: {pred_json}")

    print(f"Completed successfully.\nPredictions saved to: {pred_json}")

if __name__ == "__main__":
    run_speciesnet()
