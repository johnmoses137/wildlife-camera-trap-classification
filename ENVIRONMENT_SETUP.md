# Cluster Environment Setup (Reference Only)

⚠️ **IMPORTANT DISCLAIMER**

This document captures the **complete Python environment** used on an HPC/cluster  
to develop and run the **SpeciesNet Wildlife Camera Trap Classification System**.

- This is **NOT** the dependency list required to run this repository  
- Most users should refer to **`requirements.txt`** instead  
- This file exists for **personal reference, reproducibility, and debugging**

---

## Relationship to `requirements.txt`

- `requirements.txt` → minimal, portable, project-level dependencies  
- `ENVIRONMENT_SETUP.md` → full cluster environment snapshot  

Not all packages listed below are required for inference or data processing.

---

## Environment Overview

- **Environment type:** Python virtual environment (`venv`)
- **Execution context:** HPC / cluster (SLURM-based)
- **Primary use cases:**
  - SpeciesNet inference
  - Large-scale image processing
  - Multi-animal detection workflows
  - Interactive analysis (Jupyter)

---

## Python

- **Version:** Python 3.x
- **Environment created using:** `python -m venv`

---

## Core Model and Deep Learning

- **speciesnet** — Wildlife image classification model  
- **torch** — Core deep learning framework with GPU support  
- **torchvision** — Vision utilities and pretrained models  
- **torchaudio** — Audio utilities installed alongside torch  
- **onnx** — Neural network exchange format  
- **onnx2torch** — Convert ONNX models to PyTorch  
- **triton** — GPU kernel compilation support for PyTorch  

---

## GPU / CUDA Runtime Libraries (User Space)

> System GPU drivers must already be installed.

- nvidia-cublas-cu11 / cu12  
- nvidia-cuda-runtime-cu11 / cu12  
- nvidia-cuda-nvrtc-cu11 / cu12  
- nvidia-cuda-cupti-cu11 / cu12  
- nvidia-cudnn-cu11 / cu12  
- nvidia-cufft-cu11 / cu12  
- nvidia-curand-cu11 / cu12  
- nvidia-cusolver-cu11 / cu12  
- nvidia-cusparse-cu11 / cu12  
- nvidia-cusparselt-cu12  
- nvidia-nccl-cu11 / cu12  
- nvidia-nvjitlink-cu12  
- nvidia-nvtx-cu11 / cu12  

---

## Image Processing and Metadata

- **opencv-python**, **opencv-python-headless** — Image loading, resizing, annotation  
- **pillow**, **pillow_heif** — Image file handling (including HEIC)  
- **filetype** — Automatic file type detection  
- **ExifRead** — EXIF metadata extraction  

---

## Object Detection and Computer Vision

- **ultralytics** — YOLO-based object detection framework  
- **yolov5** — YOLOv5 model support  
- **sahi** — Sliced inference for large images  
- **pybboxes** — Bounding box format conversions  
- **shapely** — Geometric and spatial operations  
- **thop**, **ultralytics-thop** — Model complexity and FLOPs calculation  

---

## Data Handling and Machine Learning

- **numpy** — Numerical computing  
- **pandas** — Tabular data processing  
- **polars**, **polars-runtime-32** — High-performance DataFrames  
- **scikit-learn** — Machine learning utilities  
- **scipy** — Scientific computing  
- **joblib** — Lightweight parallel processing  
- **threadpoolctl** — CPU thread control  
- **pytz**, **tzdata** — Timezone handling  

---

## Visualization

- **matplotlib**, **matplotlib-inline** — Plotting and notebook visualization  
- **seaborn** — Statistical visualization  
- **squarify** — Treemap visualizations  
- **tensorboard**, **tensorboard-data-server** — Model visualization  
- **contourpy**, **cycler** — Matplotlib internals  
- **fonttools** — Font manipulation  
- **kiwisolver** — Constraint solver for matplotlib  

---

## Jupyter and Interactive Computing

- **ipython** — Enhanced interactive shell  
- **ipykernel** — Jupyter kernel  
- **jupyter_client**, **jupyter_core** — Jupyter protocol support  
- **comm**, **traitlets** — Kernel/frontend communication  
- **prompt_toolkit** — Interactive command-line support  
- **pyzmq** — ZeroMQ bindings  
- **tornado**, **nest-asyncio** — Async networking and event loops  

---

## IPython Introspection and Debugging

- **asttokens** — AST source annotations  
- **executing** — Runtime execution inspection  
- **stack-data** — Stack frame inspection  
- **pure_eval** — Safe expression evaluation  
- **debugpy** — Python debugger  
- **jedi**, **parso** — Autocompletion and static analysis  
- **pexpect**, **ptyprocess** — Interactive process control  
- **decorator** — Decorator utilities  
- **Pygments** — Syntax highlighting  

---

## Cloud, Data Access, and Version Control

- **boto3**, **botocore**, **s3transfer** — AWS SDK  
- **huggingface-hub** — Model downloads  
- **kagglehub** — Kaggle dataset access  
- **roboflow** — Dataset management  
- **GitPython**, **gitdb**, **smmap** — Git access  
- **cloudpathlib**, **fsspec** — Unified filesystem abstraction  

---

## Web and HTTP Libraries

- **requests** — HTTP client  
- **urllib3** — Connection pooling  
- **requests-toolbelt** — Advanced HTTP utilities  
- **certifi** — CA certificates  
- **charset-normalizer**, **idna** — Encoding and IDN support  

---

## Command Line and Utilities

- **click** — CLI creation toolkit  
- **fire** — Automatic CLI generation  
- **tqdm** — Progress bars  
- **humanfriendly** — Human-readable I/O  
- **termcolor** — ANSI color formatting  
- **terminaltables** — ASCII tables  

---

## Data Serialization and Configuration

- **PyYAML** — YAML parsing  
- **python-dotenv** — Environment variable management  
- **protobuf** — Google data interchange format  

---

## System and Process Utilities

- **psutil** — System and process monitoring  
- **platformdirs** — Platform-specific directories  
- **py-cpuinfo** — CPU information  

---

## Geospatial and Location

- **reverse_geocoder** — Offline reverse geocoding  

---

## Mathematical and Symbolic Computation

- **sympy** — Symbolic mathematics  
- **mpmath** — Arbitrary-precision arithmetic  
- **networkx** — Graph algorithms  
- **pyparsing** — Parsing utilities  

---

## Machine Learning Utilities

- **absl-py** — Abseil utilities (TensorFlow dependency)  
- **grpcio** — RPC framework  
- **Markdown** — Markdown processing  
- **Werkzeug** — WSGI utilities  

---

## Python Package Management and Metadata

- **pip** — Package installer  
- **setuptools** — Build tools  
- **packaging** — Version handling  
- **filelock** — File locking  
- **importlib_metadata**, **importlib_resources**, **zipp** — Package metadata  
- **typing_extensions** — Backported typing  

---

## Template and Markup Processing

- **Jinja2** — Templating engine  
- **MarkupSafe** — Safe HTML/XML strings  

---

## Date and Time Handling

- **python-dateutil** — Datetime extensions  

---

## Miscellaneous Utilities

- **six** — Python 2/3 compatibility  
- **exceptiongroup** — Exception groups backport  
- **wcwidth** — Unicode width calculation  
- **jmespath** — JSON query language  

---

## Summary

- **Total packages:** 145  
- **Environment supports:**
  - Wildlife image classification with SpeciesNet  
  - Multi-animal object detection  
  - Large-scale image processing  
  - Model evaluation and experimentation  
  - Cloud-based data workflows  
  - Interactive Jupyter analysis  

---

## Installation Instructions

### Prerequisites

- Python 3.8+  
- NVIDIA GPU with CUDA-capable drivers  
- CUDA 11.8 or 12.x  
- ~5 GB free disk space  

---

### Method 1: Install from `requirements.txt` (Recommended)

```bash
python3 -m venv ~/speciesnet_env
source ~/speciesnet_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip list
