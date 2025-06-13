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


def mask_polygon_from_tif(
    fullres_path: str,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    scalefactors: list[float],
    level: int = 0,
    plot: bool = True
) -> tuple[np.ndarray, np.ndarray]:

    # load image
    img = tifffile.imread(fullres_path, is_ome=False, level=level)

    # rescale coords
    factor = scalefactors[level]
    pts = np.column_stack((np.array(x_coords)/factor, np.array(y_coords)/factor))
    pts = pts.astype(np.int32).reshape(-1,1,2)

    # build & apply mask
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    masked = np.where(mask[...,None] if img.ndim==3 else mask>0, img, 0)

    # optional plotting
    if plot:
        fig, axes = plt.subplots(1,3,figsize=(15,5))
        axes[0].imshow(img,    cmap='binary'); axes[0].set_title('Original'); axes[0].axis('off')
        axes[1].imshow(mask,   cmap='binary'); axes[1].set_title('Mask');     axes[1].axis('off')
        axes[2].imshow(masked, cmap='binary'); axes[2].set_title('Masked');   axes[2].axis('off')
        plt.tight_layout(); plt.show()

    return mask, masked


def crop_polygon_to_overlapping_squares(polygon, square_size, overlap_size=40/0.2125):
    """
    Crops a polygon into overlapping square regions of a given size.

    Args:
        polygon (Polygon): The input polygon to crop.
        square_size (float): The base size of each square region (side length).
        overlap_size (float): The amount of overlap in um (default: 20).

    Returns:
        List[Polygon]: List of overlapping square polygons that fully contain the polygon.
    """
    # Get bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds

    # Generate grid of overlapping squares
    squares = []
    for x in range(int(minx), int(maxx), square_size):
        for y in range(int(miny), int(maxy), square_size):
            # Create expanded square with overlap
            # First vertex (x, y) remains the same
            # Other vertices are expanded by overlap_size
            square = box(
                x,                          # minx (unchanged)
                y,                          # miny (unchanged)
                x + square_size + overlap_size,  # maxx (expanded)
                y + square_size + overlap_size   # maxy (expanded)
            )
            # Check if the square intersects the polygon
            if polygon.intersects(square):
                squares.append(square)
    
    return squares

def plot_polygon_and_squares(
    polygon, 
    squares, 
    title="Non-Overlapping Squares Enclosing Region",
    polygon_label="Polygon",
    square_label="Square",
    square_edge="red",
    square_alpha=0.3
):

    fig, ax = plt.subplots()
    # plot the polygon outline
    x, y = polygon.exterior.xy
    ax.plot(x, y, 'k-', label=polygon_label)

    # fill squares (only label the first for legend)
    for i, sq in enumerate(squares):
        sx, sy = sq.exterior.xy
        ax.fill(sx, sy,
                edgecolor=square_edge,
                alpha=square_alpha,
                label=square_label if i == 0 else None)

    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ax.legend(loc="upper left")
    plt.show()
    
# Define a function to crop the image based on a shapely Polygon
def crop_region(image, poly):
    """
    Crops the rectangular bounding box of a polygon from an image.

    Parameters:
        image (ndarray): The original image to crop.
        poly (Polygon): A shapely Polygon object defining the region.

    Returns:
        ndarray: The cropped image region.
    """
    # Get the bounding box of the polygon
    min_x, min_y, max_x, max_y = map(int, poly.bounds)
    
    # Crop the image using the bounding box
    cropped_image = image[min_y:max_y, min_x:max_x]
    
    return cropped_image