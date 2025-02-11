# Xenium Re-Segmentation Pipeline Using Cellpose
This repository contains a pipeline for re-segmentation of Xenium images using Cellpose. The pipeline includes the following steps:

1. Cropping Xenium Images (Optional)
Extracts smaller patches based on the Region of Interest (ROI) for efficient processing.

2. Cellpose Model Initialization & Fine-Tuning (Optional)
Loads a pre-trained Cellpose model and allows fine-tuning for dataset-specific segmentation.

3. Segmentation with Cellpose
Applies the Cellpose model to segment cells in cropped Xenium images.

4. Merging Segmented Patches
Reconstructs the full image by merging segmented patches back into their original spatial context.

5. Assigning Transcripts to Cells
Maps spatial transcriptomic data to the segmented cells for downstream analysis.

## Installation & Dependencies

Ensure you have the following dependencies installed:

```python
pip install cellpose numpy matplotlib tifffile
```

## Usage
1. Prepare Xenium images
Organize images in a structured format before processing.

2. Run the segmentation pipeline
Execute the provided scripts sequentially or as a complete workflow.

3. Adjust parameters as needed
Modify hyperparameters based on dataset characteristics.