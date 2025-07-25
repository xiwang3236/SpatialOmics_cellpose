# SpatialOmics Cellpose Resegmentation Pipeline

A comprehensive pipeline for resegmentation of Xenium spatial omics images using Cellpose, designed for high-throughput analysis of spatial transcriptomics data.

## 🚀 Quick Start

1. **Main Pipeline**: [resegmentation_pipeline.ipynb](resegmentation_pipeline.ipynb) - Complete end-to-end resegmentation workflow
2. **Installation**: `pip install -r requirements.txt` or `conda env create -f environment.yml`

## 📁 Repository Structure

```
SpatialOmics_cellpose/
├── 📊 data5k_r2/                    # Xenium output data (5k genes, round 2)
│   └── output-XETG00245__*/         # Sample-specific data folders
├── 📊 data5k/                       # Xenium output data (5k genes, round 1)
├── 🔧 SRC_RESEG/                    # Core resegmentation modules
│   ├── [crop_image.py](SRC_RESEG/crop_image.py)           # Image cropping utilities
│   ├── [segmentation_cp.py](SRC_RESEG/segmentation_cp.py) # Cellpose segmentation
│   ├── [assign_transcripts.py](SRC_RESEG/assign_transcripts.py) # Transcript assignment
│   └── [merge.py](SRC_RESEG/merge.py)                     # Data merging utilities
├── 🛠️ utils/                        # Utility notebooks and tools
│   ├── [crop_from_xenium.ipynb](utils/crop_from_xenium.ipynb)
│   ├── [comparision_matrix.ipynb](utils/comparision_matrix.ipynb)
│   └── [convert.ipynb](utils/convert.ipynb)
├── 🤖 models/                       # Trained Cellpose models
├── ⚙️ configs/                      # Configuration files
├── 📈 results/                      # Pipeline outputs
│   ├── cropped_image/               # Cropped image patches
│   ├── gene_expression_matrix/      # Gene expression matrices
│   └── segmentation/                # Segmentation masks
├── [resegmentation_pipeline.ipynb](resegmentation_pipeline.ipynb)  # Main pipeline
├── [requirements.txt](requirements.txt)                    # Python dependencies
└── [environment.yml](environment.yml)                      # Conda environment
```

## 🔧 Core Components

### Main Pipeline
- **[resegmentation_pipeline.ipynb](resegmentation_pipeline.ipynb)**: Complete resegmentation workflow including ROI definition, image cropping, and Cellpose segmentation

### Core Modules (`SRC_RESEG/`)
- **[crop_image.py](SRC_RESEG/crop_image.py)**: Handles image cropping, polygon masking, and patch generation
- **[segmentation_cp.py](SRC_RESEG/segmentation_cp.py)**: Cellpose-based cell segmentation with custom model support
- **[assign_transcripts.py](SRC_RESEG/assign_transcripts.py)**: Assigns transcript coordinates to segmented cells
- **[merge.py](SRC_RESEG/merge.py)**: Merges multiple datasets and results

### Data Organization
- **data5k_r2/**: Latest Xenium data (5k genes, round 2 experiments)
- **data5k/**: Previous Xenium data (5k genes, round 1 experiments)
- **models/**: Pre-trained Cellpose models for specific tissue types

### Utilities (`utils/`)
- **[crop_from_xenium.ipynb](utils/crop_from_xenium.ipynb)**: Xenium-specific cropping utilities
- **[comparision_matrix.ipynb](utils/comparision_matrix.ipynb)**: Analysis and comparison tools
- **[convert.ipynb](utils/convert.ipynb)**: Data format conversion utilities

## 🎯 Workflow

1. **Data Preparation**: Load Xenium output data from `data5k_r2/`
2. **ROI Definition**: Define regions of interest using coordinate files
3. **Image Cropping**: Generate overlapping patches for segmentation
4. **Cell Segmentation**: Apply Cellpose models for cell boundary detection
5. **Transcript Assignment**: Map transcript coordinates to segmented cells
6. **Data Integration**: Merge results and generate expression matrices

## 📋 Requirements

- Python 3.11+
- Cellpose 4.0+
- PyTorch with CUDA support (recommended)
- See [requirements.txt](requirements.txt) for complete dependencies

## 🔗 Key Files

- **Main Pipeline**: [resegmentation_pipeline.ipynb](resegmentation_pipeline.ipynb)
- **Core Modules**: [SRC_RESEG/](SRC_RESEG/)
- **Dependencies**: [requirements.txt](requirements.txt), [environment.yml](environment.yml)
- **Data**: [data5k_r2/](data5k_r2/), [data5k/](data5k/)
- **Results**: [results/](results/)
