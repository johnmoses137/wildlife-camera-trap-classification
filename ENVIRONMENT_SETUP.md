# Cluster Environment Setup (Reference Only)

⚠️ IMPORTANT DISCLAIMER

This document describes the **complete Python environment** used on an HPC/cluster
to develop and run the SpeciesNet wildlife classification pipeline.

❗ Not all packages listed here are required to run this repository.
❗ This is NOT a replacement for `requirements.txt`.

- `requirements.txt` → minimal, portable, project-level dependencies
- `environment_setup.md` → full cluster environment snapshot (personal reference)

This file exists for:
- Recreating the same environment on similar clusters
- Debugging dependency issues
- Long-term personal reference


## Relationship to requirements.txt

- `requirements.txt` lists **only the minimal Python packages directly imported**
  by this repository.
- This document lists **all packages installed in the cluster environment**,
  including deep learning frameworks, GPU runtimes, experimentation tools,
  cloud utilities, and Jupyter support.

Not all packages listed here are required for inference or data processing.


## Installed Packages (Full Environment Snapshot)
fonttools
  Font manipulation and analysis

kiwisolver
  Fast constraint solver for matplotlib


============================================================
JUPYTER AND INTERACTIVE COMPUTING
============================================================
ipython
  Enhanced interactive Python shell

ipykernel
  IPython kernel for Jupyter

jupyter_client
  Jupyter protocol client APIs

jupyter_core
  Core Jupyter functionality

comm
  Communication between kernel and frontend

traitlets
  Configuration system for Jupyter

prompt_toolkit
  Library for building interactive command lines

pyzmq
  Python bindings for ZeroMQ messaging

tornado
  Asynchronous networking library

nest-asyncio
  Patch asyncio to allow nested event loops

matplotlib-inline
  Inline Matplotlib backend for Jupyter


============================================================
IPYTHON INTROSPECTION AND DEBUGGING
============================================================
asttokens
  Annotate Python AST with source code positions

executing
  Get information about what Python is currently executing

stack-data
  Extract data from Python stack frames

pure_eval
  Safely evaluate Python expressions

debugpy
  Python debugger for Visual Studio Code

jedi
  Autocompletion and static analysis for Python

parso
  Python parser used by Jedi

pexpect
  Control interactive applications

ptyprocess
  Run processes in pseudo-terminals

decorator
  Simplify usage of decorators

Pygments
  Syntax highlighting library


============================================================
CLOUD, DATA ACCESS, AND VERSION CONTROL
============================================================
boto3
  AWS SDK for Python

botocore
  Low-level AWS service access

s3transfer
  Amazon S3 transfer manager

huggingface-hub
  Model and artifact downloads from Hugging Face

kagglehub
  Kaggle dataset access

roboflow
  Dataset management and computer vision tooling

GitPython
  Programmatic Git repository access

gitdb
  Git object database

smmap
  Pure Python implementation of memory-mapped files

cloudpathlib
  Unified filesystem access (local and cloud)

fsspec
  Filesystem abstraction layer


============================================================
WEB AND HTTP LIBRARIES
============================================================
requests
  HTTP library for making web requests

urllib3
  HTTP client with connection pooling

requests-toolbelt
  Utility belt for advanced requests usage

certifi
  Mozilla's CA certificate bundle

charset-normalizer
  Character encoding detection

idna
  Internationalized Domain Names support


============================================================
COMMAND LINE AND UTILITIES
============================================================
click
  Command line interface creation toolkit

fire
  Automatic command line interfaces

tqdm
  Progress bar for loops and CLI

humanfriendly
  Human-friendly input/output

termcolor
  ANSI color formatting for terminal output

terminaltables
  ASCII tables for terminal output


============================================================
DATA SERIALIZATION AND CONFIGURATION
============================================================
PyYAML
  YAML parser and emitter

python-dotenv
  Environment variable management from .env files

protobuf
  Google's data interchange format


============================================================
SYSTEM AND PROCESS UTILITIES
============================================================
psutil
  System and process monitoring

platformdirs
  Platform-specific system directories

py-cpuinfo
  CPU information retrieval


============================================================
GEOSPATIAL AND LOCATION
============================================================
reverse_geocoder
  Reverse geocoding without external services


============================================================
MATHEMATICAL AND SYMBOLIC COMPUTATION
============================================================
sympy
  Symbolic mathematics

mpmath
  Arbitrary-precision floating-point arithmetic

networkx
  Network and graph algorithms

pyparsing
  Parsing library


============================================================
MACHINE LEARNING UTILITIES
============================================================
absl-py
  Abseil Python common libraries (used by TensorFlow)

grpcio
  HTTP/2-based RPC framework

Markdown
  Python implementation of Markdown

Werkzeug
  WSGI utility library for Python


============================================================
PYTHON PACKAGE MANAGEMENT AND METADATA
============================================================
pip
  Python package installer

setuptools
  Python package build and distribution tools

packaging
  Core packaging utilities and version handling

filelock
  Platform-independent file locking

importlib_metadata
  Read metadata from Python packages

importlib_resources
  Read resources from Python packages

zipp
  Backport of pathlib-compatible object wrapper

typing_extensions
  Backported type hints for older Python versions


============================================================
TEMPLATE AND MARKUP PROCESSING
============================================================
Jinja2
  Templating engine for Python

MarkupSafe
  Safe string handling for HTML/XML


============================================================
DATE AND TIME HANDLING
============================================================
python-dateutil
  Extensions to Python datetime module


============================================================
MISCELLANEOUS UTILITIES
============================================================
six
  Python 2 and 3 compatibility utilities

exceptiongroup
  Backport of exception groups

wcwidth
  Measure displayed width of unicode strings

jmespath
  JSON query language


============================================================
SUMMARY
============================================================
Total packages: 145

This environment is a complete machine learning and computer vision pipeline with:
- Deep learning: PyTorch with CUDA 11 and 12 support
- Computer vision: OpenCV, Pillow, Ultralytics YOLO, YOLOv5, SAHI
- Data science: NumPy, Pandas, Polars, scikit-learn, SciPy
- Visualization: Matplotlib, Seaborn, TensorBoard
- Cloud integration: AWS (boto3), Hugging Face, Kaggle, Roboflow
- Development tools: Jupyter, IPython with debugging support
- Geospatial: Reverse geocoding capabilities
- CLI tools: Click, Fire, tqdm with progress bars

Environment is ready for:
- Wildlife image classification with SpeciesNet
- Object detection and tracking
- Large-scale image processing
- Model training and evaluation
- Cloud-based data pipelines
- Interactive analysis in Jupyter notebooks


============================================================
INSTALLATION INSTRUCTIONS
============================================================

PREREQUISITES
-------------
Before setting up this environment, ensure you have:
- Python 3.8 or higher installed
- NVIDIA GPU with CUDA-capable drivers (for GPU acceleration)
- CUDA 11.8 or 12.x drivers installed on the system
- At least 5GB of free disk space
- Internet connection for downloading packages


METHOD 1: INSTALL FROM REQUIREMENTS.TXT (RECOMMENDED)
------------------------------------------------------
This is the fastest way to recreate this exact environment.

1. Create a new virtual environment:
   python3 -m venv ~/speciesnet_env

2. Activate the environment:
   source ~/speciesnet_env/bin/activate

3. Upgrade pip:
   pip install --upgrade pip

4. Install all packages from requirements file:
   pip install -r requirements.txt

5. Verify installation:
   pip list


METHOD 2: CREATE ENVIRONMENT FROM SCRATCH
------------------------------------------
Step-by-step manual setup if you want to understand the process.

1. Create virtual environment:
   python3 -m venv ~/speciesnet_env

2. Activate the environment:
   source ~/speciesnet_env/bin/activate

3. Upgrade pip:
   pip install --upgrade pip

4. Install PyTorch with CUDA support FIRST (critical for dependencies):
   pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

5. Install the SpeciesNet package:
   pip install speciesnet==5.0.0

6. Install all remaining packages:
   pip install numpy pandas polars scikit-learn scipy matplotlib seaborn \
   opencv-python pillow ultralytics yolov5 sahi boto3 huggingface-hub \
   kagglehub roboflow jupyter ipython tensorboard click fire tqdm PyYAML \
   requests GitPython cloudpathlib reverse_geocoder sympy networkx \
   onnx onnx2torch ExifRead filetype pillow_heif pybboxes shapely thop \
   ultralytics-thop joblib threadpoolctl contourpy cycler fonttools \
   kiwisolver grpcio protobuf python-dotenv requests-toolbelt \
   humanfriendly termcolor terminaltables py-cpuinfo mpmath pyparsing \
   absl-py Markdown Werkzeug

7. Verify installation:
   pip list


METHOD 3: GENERATE REQUIREMENTS FILE FROM THIS ENVIRONMENT
-----------------------------------------------------------
Use this to create a requirements.txt file for sharing or backup.

1. Activate this environment:
   source ~/speciesnet_env/bin/activate

2. Generate requirements file with exact versions:
   pip freeze > requirements.txt

3. (Optional) Generate without version constraints:
   pip list --format=freeze | sed 's/==.*//' > requirements_no_versions.txt

4. The requirements.txt file can now be used with METHOD 1 to recreate
   this exact environment on another machine.


ACTIVATION AND DEACTIVATION
----------------------------
To activate the environment:
   source ~/speciesnet_env/bin/activate

To deactivate when done:
   deactivate


TESTING THE INSTALLATION
-------------------------
After installation, test that key components work:

1. Test PyTorch and CUDA:
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

2. Test SpeciesNet:
   python -c "import speciesnet; print(f'SpeciesNet: {speciesnet.__version__}')"

3. Test computer vision libraries:
   python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
   python -c "from PIL import Image; print('Pillow: OK')"

4. Test data science stack:
   python -c "import numpy, pandas, polars; print('Data libraries: OK')"


TROUBLESHOOTING
---------------
Common issues and solutions:

1. CUDA not available:
   - Verify NVIDIA drivers are installed: nvidia-smi
   - Ensure CUDA toolkit is installed
   - Reinstall PyTorch with correct CUDA version

2. Package conflicts:
   - Create a fresh virtual environment
   - Install PyTorch FIRST before other packages

3. Import errors:
   - Ensure environment is activated
   - Verify package installation: pip show <package-name>

4. Out of memory errors:
   - Reduce batch size in model configurations
   - Close other GPU-intensive applications
(py-venv-jupyter) (py-venv-jupyter) (py-venv-jupyter) (py-venv-jupyter) ls -la ~/speciesnet_env/README.md
-rw-r--r-- 1 bollaraj bollaraj 14749 Dec 16 11:19 /mnt/home/bollaraj/speciesnet_env/README.md
(py-venv-jupyter) (py-venv-jupyter) cat ~/speciesnet_env/README.md
SpeciesNet Environment (speciesnet_env)

PURPOSE
-------
This directory contains a Python virtual environment used to run the
SpeciesNet wildlife image classification model along with the complete
supporting data, vision, GPU, and analysis pipeline.

This README lists ALL libraries installed in the environment, grouped by
similar or related functionality, with a one-line explanation for each.
No installed package is omitted.


ENVIRONMENT LOCATION
--------------------
~/speciesnet_env


PYTHON
------
Python 3.x
Environment created using python venv.


============================================================
CORE MODEL AND DEEP LEARNING
============================================================
speciesnet
  Wildlife image classification model

torch
  Core deep learning framework with GPU support

torchvision
  Vision utilities and image models for PyTorch

torchaudio
  Audio utilities installed alongside torch

onnx
  Neural network model exchange format

onnx2torch
  Converts ONNX models to PyTorch format

triton
  GPU kernel compilation support for PyTorch


============================================================
GPU / CUDA RUNTIME LIBRARIES (USER SPACE)
============================================================
nvidia-cublas-cu11
nvidia-cublas-cu12
nvidia-cuda-cupti-cu11
nvidia-cuda-cupti-cu12
nvidia-cuda-nvrtc-cu11
nvidia-cuda-nvrtc-cu12
nvidia-cuda-runtime-cu11
nvidia-cuda-runtime-cu12
nvidia-cudnn-cu11
nvidia-cudnn-cu12
nvidia-cufft-cu11
nvidia-cufft-cu12
nvidia-cufile-cu12
nvidia-curand-cu11
nvidia-curand-cu12
nvidia-cusolver-cu11
nvidia-cusolver-cu12
nvidia-cusparse-cu11
nvidia-cusparse-cu12
nvidia-cusparselt-cu12
nvidia-nccl-cu11
nvidia-nccl-cu12
nvidia-nvjitlink-cu12
nvidia-nvtx-cu11
nvidia-nvtx-cu12

  CUDA runtime, math, and GPU acceleration libraries required by PyTorch.
  System GPU drivers must already be installed.


============================================================
IMAGE PROCESSING AND METADATA
============================================================
opencv-python
opencv-python-headless
  Image loading, resizing, annotation, and processing

pillow
pillow_heif
  Image file handling including HEIC format support

filetype
  Automatic file type detection

ExifRead
  EXIF metadata extraction from images


============================================================
OBJECT DETECTION AND COMPUTER VISION
============================================================
ultralytics
  YOLO-based object detection framework

yolov5
  YOLOv5 detection model support

sahi
  Sliced inference for large images

pybboxes
  Bounding box format conversions

shapely
  Geometric and spatial operations

thop
  Model complexity and FLOPs calculation

ultralytics-thop
  THOP integration for Ultralytics models


============================================================
DATA HANDLING AND MACHINE LEARNING
============================================================
numpy
  Numerical computing

pandas
  Tabular data processing

polars
polars-runtime-32
  High-performance DataFrame operations

scikit-learn
  Machine learning utilities

scipy
  Scientific computing

joblib
  Lightweight parallel processing

threadpoolctl
  CPU thread control for numerical libraries

pytz
  Timezone definitions

tzdata
  IANA timezone database


============================================================
VISUALIZATION
============================================================
matplotlib
matplotlib-inline
  Plotting and notebook visualization

seaborn
  Statistical visualization

squarify
  Treemap visualizations

tensorboard
tensorboard-data-server
  Model training and inference visualization

contourpy
  Contour line generation for matplotlib

cycler
  Composable style cycles for matplotlib

fonttools
  Font manipulation and analysis

kiwisolver
  Fast constraint solver for matplotlib


============================================================
JUPYTER AND INTERACTIVE COMPUTING
============================================================
ipython
  Enhanced interactive Python shell

ipykernel
  IPython kernel for Jupyter

jupyter_client
  Jupyter protocol client APIs

jupyter_core
  Core Jupyter functionality

comm
  Communication between kernel and frontend

traitlets
  Configuration system for Jupyter

prompt_toolkit
  Library for building interactive command lines

pyzmq
  Python bindings for ZeroMQ messaging

tornado
  Asynchronous networking library

nest-asyncio
  Patch asyncio to allow nested event loops

matplotlib-inline
  Inline Matplotlib backend for Jupyter


============================================================
IPYTHON INTROSPECTION AND DEBUGGING
============================================================
asttokens
  Annotate Python AST with source code positions

executing
  Get information about what Python is currently executing

stack-data
  Extract data from Python stack frames

pure_eval
  Safely evaluate Python expressions

debugpy
  Python debugger for Visual Studio Code

jedi
  Autocompletion and static analysis for Python

parso
  Python parser used by Jedi

pexpect
  Control interactive applications

ptyprocess
  Run processes in pseudo-terminals

decorator
  Simplify usage of decorators

Pygments
  Syntax highlighting library


============================================================
CLOUD, DATA ACCESS, AND VERSION CONTROL
============================================================
boto3
  AWS SDK for Python

botocore
  Low-level AWS service access

s3transfer
  Amazon S3 transfer manager

huggingface-hub
  Model and artifact downloads from Hugging Face

kagglehub
  Kaggle dataset access

roboflow
  Dataset management and computer vision tooling

GitPython
  Programmatic Git repository access

gitdb
  Git object database

smmap
  Pure Python implementation of memory-mapped files

cloudpathlib
  Unified filesystem access (local and cloud)

fsspec
  Filesystem abstraction layer


============================================================
WEB AND HTTP LIBRARIES
============================================================
requests
  HTTP library for making web requests

urllib3
  HTTP client with connection pooling

requests-toolbelt
  Utility belt for advanced requests usage

certifi
  Mozilla's CA certificate bundle

charset-normalizer
  Character encoding detection

idna
  Internationalized Domain Names support


============================================================
COMMAND LINE AND UTILITIES
============================================================
click
  Command line interface creation toolkit

fire
  Automatic command line interfaces

tqdm
  Progress bar for loops and CLI

humanfriendly
  Human-friendly input/output

termcolor
  ANSI color formatting for terminal output

terminaltables
  ASCII tables for terminal output


============================================================
DATA SERIALIZATION AND CONFIGURATION
============================================================
PyYAML
  YAML parser and emitter

python-dotenv
  Environment variable management from .env files

protobuf
  Google's data interchange format


============================================================
SYSTEM AND PROCESS UTILITIES
============================================================
psutil
  System and process monitoring

platformdirs
  Platform-specific system directories

py-cpuinfo
  CPU information retrieval


============================================================
GEOSPATIAL AND LOCATION
============================================================
reverse_geocoder
  Reverse geocoding without external services


============================================================
MATHEMATICAL AND SYMBOLIC COMPUTATION
============================================================
sympy
  Symbolic mathematics

mpmath
  Arbitrary-precision floating-point arithmetic

networkx
  Network and graph algorithms

pyparsing
  Parsing library


============================================================
MACHINE LEARNING UTILITIES
============================================================
absl-py
  Abseil Python common libraries (used by TensorFlow)

grpcio
  HTTP/2-based RPC framework

Markdown
  Python implementation of Markdown

Werkzeug
  WSGI utility library for Python


============================================================
PYTHON PACKAGE MANAGEMENT AND METADATA
============================================================
pip
  Python package installer

setuptools
  Python package build and distribution tools

packaging
  Core packaging utilities and version handling

filelock
  Platform-independent file locking

importlib_metadata
  Read metadata from Python packages

importlib_resources
  Read resources from Python packages

zipp
  Backport of pathlib-compatible object wrapper

typing_extensions
  Backported type hints for older Python versions


============================================================
TEMPLATE AND MARKUP PROCESSING
============================================================
Jinja2
  Templating engine for Python

MarkupSafe
  Safe string handling for HTML/XML


============================================================
DATE AND TIME HANDLING
============================================================
python-dateutil
  Extensions to Python datetime module


============================================================
MISCELLANEOUS UTILITIES
============================================================
six
  Python 2 and 3 compatibility utilities

exceptiongroup
  Backport of exception groups

wcwidth
  Measure displayed width of unicode strings

jmespath
  JSON query language


============================================================
SUMMARY
============================================================
Total packages: 145

This environment is a complete machine learning and computer vision pipeline with:
- Deep learning: PyTorch with CUDA 11 and 12 support
- Computer vision: OpenCV, Pillow, Ultralytics YOLO, YOLOv5, SAHI
- Data science: NumPy, Pandas, Polars, scikit-learn, SciPy
- Visualization: Matplotlib, Seaborn, TensorBoard
- Cloud integration: AWS (boto3), Hugging Face, Kaggle, Roboflow
- Development tools: Jupyter, IPython with debugging support
- Geospatial: Reverse geocoding capabilities
- CLI tools: Click, Fire, tqdm with progress bars

Environment is ready for:
- Wildlife image classification with SpeciesNet
- Object detection and tracking
- Large-scale image processing
- Model training and evaluation
- Cloud-based data pipelines
- Interactive analysis in Jupyter notebooks


============================================================
INSTALLATION INSTRUCTIONS
============================================================

PREREQUISITES
-------------
Before setting up this environment, ensure you have:
- Python 3.8 or higher installed
- NVIDIA GPU with CUDA-capable drivers (for GPU acceleration)
- CUDA 11.8 or 12.x drivers installed on the system
- At least 5GB of free disk space
- Internet connection for downloading packages


METHOD 1: INSTALL FROM REQUIREMENTS.TXT (RECOMMENDED)
------------------------------------------------------
This is the fastest way to recreate this exact environment.

1. Create a new virtual environment:
   python3 -m venv ~/speciesnet_env

2. Activate the environment:
   source ~/speciesnet_env/bin/activate

3. Upgrade pip:
   pip install --upgrade pip

4. Install all packages from requirements file:
   pip install -r requirements.txt

5. Verify installation:
   pip list


METHOD 2: CREATE ENVIRONMENT FROM SCRATCH
------------------------------------------
Step-by-step manual setup if you want to understand the process.

1. Create virtual environment:
   python3 -m venv ~/speciesnet_env

2. Activate the environment:
   source ~/speciesnet_env/bin/activate

3. Upgrade pip:
   pip install --upgrade pip

4. Install PyTorch with CUDA support FIRST (critical for dependencies):
   pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118

5. Install the SpeciesNet package:
   pip install speciesnet==5.0.0

6. Install all remaining packages:
   pip install numpy pandas polars scikit-learn scipy matplotlib seaborn \
   opencv-python pillow ultralytics yolov5 sahi boto3 huggingface-hub \
   kagglehub roboflow jupyter ipython tensorboard click fire tqdm PyYAML \
   requests GitPython cloudpathlib reverse_geocoder sympy networkx \
   onnx onnx2torch ExifRead filetype pillow_heif pybboxes shapely thop \
   ultralytics-thop joblib threadpoolctl contourpy cycler fonttools \
   kiwisolver grpcio protobuf python-dotenv requests-toolbelt \
   humanfriendly termcolor terminaltables py-cpuinfo mpmath pyparsing \
   absl-py Markdown Werkzeug

7. Verify installation:
   pip list


METHOD 3: GENERATE REQUIREMENTS FILE FROM THIS ENVIRONMENT
-----------------------------------------------------------
Use this to create a requirements.txt file for sharing or backup.

1. Activate this environment:
   source ~/speciesnet_env/bin/activate

2. Generate requirements file with exact versions:
   pip freeze > requirements.txt

3. (Optional) Generate without version constraints:
   pip list --format=freeze | sed 's/==.*//' > requirements_no_versions.txt

4. The requirements.txt file can now be used with METHOD 1 to recreate
   this exact environment on another machine.


ACTIVATION AND DEACTIVATION
----------------------------
To activate the environment:
   source ~/speciesnet_env/bin/activate

To deactivate when done:
   deactivate


TESTING THE INSTALLATION
-------------------------
After installation, test that key components work:

1. Test PyTorch and CUDA:
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

2. Test SpeciesNet:
   python -c "import speciesnet; print(f'SpeciesNet: {speciesnet.__version__}')"

3. Test computer vision libraries:
   python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
   python -c "from PIL import Image; print('Pillow: OK')"

4. Test data science stack:
   python -c "import numpy, pandas, polars; print('Data libraries: OK')"


TROUBLESHOOTING
---------------
Common issues and solutions:

1. CUDA not available:
   - Verify NVIDIA drivers are installed: nvidia-smi
   - Ensure CUDA toolkit is installed
   - Reinstall PyTorch with correct CUDA version

2. Package conflicts:
   - Create a fresh virtual environment
   - Install PyTorch FIRST before other packages

3. Import errors:
   - Ensure environment is activated
   - Verify package installation: pip show <package-name>

4. Out of memory errors:
   - Reduce batch size in model configurations
   - Close other GPU-intensive applications
(py-venv-jupyter) (py-venv-jupyter) 
