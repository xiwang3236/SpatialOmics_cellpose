# SPATIALOMICS_CELLPOSE
**On Construction**

This repository contains a pipeline for re-segmentation of Xenium images using Cellpose. 

## Repository Structure

SPATIALOMICS_CELLPOSE
├── configs/
├── data/
│   ├── output-XETG00245__0034182__1835od__20240613__1...
│   └── output-XETG00245__0034182__1835os__20240613__19...
├── models/
│   ├── HILT_Alexis_1028_5
│   └── HILT_Alexis_1113
├── results/
│   ├── cropped_image/
│   ├── gene_expression_matrix/
│   └── segmenatation/
├── src/
│   ├── [assign_transcripts.ipynb](src/assign_transcripts.ipynb)
│   ├── [crop_image.ipynb](src/crop_image.ipynb)
│   ├── [merge.ipynb](src/merge.ipynb)
│   ├── [segmentation.ipynb](src/segmentation.ipynb)
│   └── [train_cellpose.ipynb](src/train_cellpose.ipynb)
├── .gitignore
├── README.md
└── requirements.txt


## Folder Descriptions

- **configs/**  


- **data/**  
  Holds Xenium original output data for the further analysis. Subdirectories named `output-XETG00245__...` that contain raw images, transcript coordinates, or other experimental outputs.

- **models/**  
  Stores Cellpose fine-tuned model files (e.g., `HILT_Alexis_1028_5`, `HILT_Alexis_1113`).

- **results/**  
  Output folder containing the results of  analysis pipelines.

  - **cropped_image/**: Contains cropped image files generated by the cropping process.  
  - **gene_expression_matrix/**: Contains CSV/TSV/other matrices that summarize gene expression counts.  
  - **segmentation/**: Contains segmentation output masks, labels, or other results from the cell segmentation process.

- **src/**  
  Contains the core Jupyter notebooks for your workflow:

  - **assign_transcripts.ipynb**  
    Matches or “assigns” transcript reads to specific segmented cells, mapping transcript coordinates to cell labels.
  - **crop_image.ipynb**  
    Performs image cropping or any preprocessing needed before segmentation or other analysis steps.
  - **merge.ipynb**  
    Merges multiple datasets or results—for instance, combining multiple fields of view or multiple gene expression outputs into a single table.
  - **segmentation.ipynb**  
    Executes the Cellpose segmentation process on images, potentially loading models from the `models/` folder.
  - **train_cellpose.ipynb**  
    Demonstrates how to train or fine-tune a Cellpose model using your custom data.

- **requirements.txt**  
  Specifies Python dependencies needed to run the notebooks and scripts
