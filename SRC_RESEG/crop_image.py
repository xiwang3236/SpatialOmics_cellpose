import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from shapely.geometry import Polygon, box
import numpy as np
import cv2

def plot_roi(filepath):
    # Read coordinates and create plot
    coords = pd.read_csv(filepath)
    
    plt.figure(figsize=(6, 6))
    plt.plot(coords['X'], coords['Y'], 'b-o', markersize=2, label='Polygon ROI')
    plt.plot([coords['X'].iloc[0], coords['X'].iloc[-1]], 
            [coords['Y'].iloc[0], coords['Y'].iloc[-1]], 'b-')
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title("Polygon ROI")
    plt.legend()
    plt.grid(True)
    plt.show()

def create_masked_image(image, x_coords, y_coords, pixelsize):
    """
    Create a masked image using polygon coordinates.
    
    Args:
        image: Input image array
        x_coords: X coordinates of polygon
        y_coords: Y coordinates of polygon
        pixelsize: Scaling factor for coordinates
    
    Returns:
        tuple: (masked_image, mask)
    """
    # Rescale coordinates
    x_rescaled = x_coords / pixelsize
    y_rescaled = y_coords / pixelsize
    
    # Create points for polygon
    points = np.column_stack((x_rescaled, y_rescaled)).astype(np.int32)
    points = points.reshape((-1, 1, 2))
    
    # Create and fill mask
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], color=255)
    
    # Apply mask
    mask_bool = mask > 0
    masked_image = np.zeros_like(image)
    masked_image[mask_bool] = image[mask_bool]
    
    return masked_image, mask

def visualize_masking(original_image, mask, masked_image):
    """
    Visualize original, mask and masked images side by side.
    
    Args:
        original_image: Original input image
        mask: Binary mask
        masked_image: Result of masking
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(original_image, cmap='binary')
    plt.title('Original Image')
    plt.axis('scaled')
    
    plt.subplot(132)
    plt.imshow(mask, cmap='binary')
    plt.title('Rescaled Mask')
    plt.axis('scaled')
    
    plt.subplot(133)
    plt.imshow(masked_image, cmap='binary')
    plt.title('Masked Image')
    plt.axis('scaled')
    
    plt.tight_layout()
    plt.show()

# Usage example:
# masked_image, mask = create_masked_image(image_chanel_0, x_coords, y_coords, pixelsize)
# visualize_masking(image_chanel_0, mask, masked_image)

