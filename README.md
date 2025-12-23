# Wildlife Camera Trap Classification System

A two-pass workflow for comprehensive wildlife camera trap image analysis using SpeciesNet machine learning model to identify multiple animals per image.

## Project Overview

This project implements a novel two-pass approach to camera trap image classification that addresses a key limitation of the SpeciesNet model: while it can detect multiple animals in a single image, it only classifies the highest-confidence detection. Our solution processes images twice—first on originals, then on cropped secondary detections—to ensure all animals are identified.

**Academic Context:** Master's degree project developed in collaboration with biology department for wildlife research applications.

## Key Innovation

**Problem:** SpeciesNet detects multiple animals but only classifies one per image  
**Solution:** Crop secondary detections and re-classify them separately  
**Result:** Complete identification of all animals in camera trap footage

## Workflow Structure

### Phase 1: Original Image Processing
Processes full-resolution camera trap images through SpeciesNet and creates comprehensive outputs including predictions, annotations, species groupings, and temporal clustering.

### Phase 2: Secondary Detection Processing  
Extracts secondary animals (detections beyond the first), crops them, re-classifies via SpeciesNet, and merges results with original dataset while preserving temporal and spatial metadata.

## Repository Structure
```
├── phase1_original_images/
│   ├── run_speciesnet_model.py           # Run SpeciesNet on original images
│   └── process_predictions_steps2to7.py  # Complete processing pipeline
├── phase2_secondary_detections/
│   ├── crop_secondary_detections.py      # Extract secondary detections
│   ├── run_speciesnet_model.py           # Re-classify cropped images
│   ├── process_cropped_predictions.py    # Process crop predictions
│   └── merge_with_original_dataset.py    # Merge with Phase 1 results
└── README.md
```

## Requirements

- Python 3.8+
- SpeciesNet environment installed at `~/speciesnet_env/`
- Required packages: pandas, opencv-python, matplotlib, exifread, tqdm, squarify

## Quick Start

### Phase 1: Process Original Images
```bash
# Step 1: Run SpeciesNet
cd phase1_original_images
python run_speciesnet_model.py

# Step 2: Process predictions
python process_predictions_steps2to7.py
```

### Phase 2: Process Secondary Detections
```bash
# Step 1: Crop secondary detections
cd phase2_secondary_detections
python crop_secondary_detections.py

# Step 2: Run SpeciesNet on crops
python run_speciesnet_model.py

# Step 3: Process cropped predictions
python process_cropped_predictions.py

# Step 4: Merge datasets
python merge_with_original_dataset.py
```

## Output Structure

Phase 1 creates:
- `predictions.csv` - Structured prediction data
- `annotated_images/` - Images with bounding boxes
- `Grouped_Images/` - Species-organized folders
- `Visualizations/` - Species distribution charts
- `meta_data_clustered.csv` - Complete dataset with temporal clustering

Phase 2 creates:
- `cropped_images/` - Secondary detection crops
- `cropped_predictions_with_metadata.csv` - Classified crops with inherited metadata
- Final merged dataset combining all detections

## Use Cases

- Wildlife occupancy modeling
- Species activity pattern analysis
- Multi-species interaction studies
- Camera trap survey analysis
- Biodiversity assessments

## Author

Created as part of master's degree research in collaboration with biology department.


