import os
import glob
import tifffile as tiff
from cellpose import denoise, models, io
from skimage import measure 
import pandas as pd

def denoise_and_segment_image(img, img_name, output_path,  cellpose_model):
    """
    Perform denoising and segmentation on a 3D image using Cellpose.

    Parameters:
    - img (numpy.ndarray): Input image array to process.
    - img_name (str): Name of the input image (for logging and saving purposes).
    - output_path (str): Base path to save the output.
    - cellpose_model: Pre-initialized Cellpose segmentation model.

    Returns:
    None
    """
    # Display input image information
    print(f"Processing file: {img_name}, shape: {img.shape}")


    # Perform segmentation
    masks, flows, styles = cellpose_model.eval(
        x=img,  # Your input image array
        batch_size=8,
        diameter=31.0,
        channels=[2,1],
        do_3D=False,
        channel_axis = None,
    )
    
    print("Mask Segmentation Down!")

    # Create and save properties DataFrame
    props = measure.regionprops_table(masks, properties=['centroid'])
    props_df = pd.DataFrame(props)
    
    # Rename the columns
    props_df = props_df.rename(columns={
        'centroid-0': 'centroid_y',
        'centroid-1': 'centroid_x'
    })

    # Print number of rows (number of detected regions)
    print(f"Number of detected regions in {img_name}: {len(props_df)}")
    num_cells = len(props_df)

    # Create props CSV filename based on the original image name
    centroids_path = output_path + '_centroids.csv'

    props_df.to_csv(centroids_path, index=False)
    # print(f"Properties saved to: {centroids_path}")

    # Save the masks as TIFF files
    io.save_masks(
        images=[img],
        masks=[masks],
        flows=[flows],
        file_names=[output_path],
        png=False,
        tif=True,
        channels=[0, 0]
    )

    io.masks_flows_to_seg(
        images=[img],  # List containing the image
        masks=[masks],            # List containing the mask
        flows=[flows],            # List containing the flows
        file_names=[output_path],  # Base filename for saving
        # diams=31.0,              # Diameter used in Cellpose

    )

    print("Segmentation output saved to 'segmented_output_seg.npy'")

    return num_cells

def process_images_in_folder(input_folder, output_folder, cellpose_model):
    """
    Process all TIF images in a folder: denoise, segment, and save the output.

    Parameters:
    - input_folder (str): Path to the folder containing input images.
    - output_folder (str): Path to save the processed output images.
    - cellpose_model: Pre-initialized Cellpose segmentation model.

    Returns:
    int: Total number of cells detected across all processed images.
    """
    i = 0

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Find all TIF files in the input folder
    file_paths = glob.glob(os.path.join(input_folder, "*.tif"))
    
    total_cells = 0
    for file_path in file_paths:
        # Read the image
        img = tiff.imread(file_path)

        # Check if image has valid dimensions
        if 0 in img.shape:
            file_name = os.path.basename(file_path)
            print(f"Skipping {file_name} due to invalid dimensions: {img.shape}")
            continue

        # Get the base file name
        file_name = os.path.basename(file_path)
        file_base = os.path.splitext(file_name)[0]
        
        # Define the output path for the current image
        output_path = os.path.join(output_folder, file_base)
        
        img_name = file_name
        # Run the denoise and segmentation function
        num_cells = denoise_and_segment_image(
            img=img,
            img_name=file_name,
            output_path=output_path,   
            cellpose_model=cellpose_model
        )
        total_cells += num_cells
        i += 1

        print(f"Output of [{img_name}] saved as: {output_path}\n")

    return total_cells