import numpy as np
import tifffile
import os
import glob

def convert_npy_to_tif_cellpose(input_folder, output_folder):
    """
    Convert Cellpose .npy files to .tif using cellpose method
    """
    os.makedirs(output_folder, exist_ok=True)
    
    npy_files = glob.glob(os.path.join(input_folder, "*_seg.npy"))
    
    if not npy_files:
        print(f"No _seg.npy files found in {input_folder}")
        return
    
    print(f"Found {len(npy_files)} .npy files to convert")
    
    for npy_file in npy_files:
        try:
            # Load using cellpose method
            dat = np.load(npy_file, allow_pickle=True).item()
            masks = dat['masks']
            
            # Get filename without extension
            basename = os.path.splitext(os.path.basename(npy_file))[0]
            output_file = os.path.join(output_folder, f"{basename}.tif")
            
            # Save with tifffile
            tifffile.imwrite(output_file, masks)
            print(f"Converted: {basename}.npy -> {basename}.tif")
            
        except Exception as e:
            print(f"Error converting {os.path.basename(npy_file)}: {str(e)}")

# Usage
input_folder = r"D:\Alexis\Projects\SpatialOmics_cellpose\results\segmenatation\Test_Lesion_DRGs_1"
output_folder = r"D:\Alexis\Projects\SpatialOmics_cellpose\results\segmenatation\Test_Lesion_DRGs_1_tif"

convert_npy_to_tif_cellpose(input_folder, output_folder)