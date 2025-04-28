# Essential imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import seaborn as sns  # For better color palettes (optional)

# Optional imports for additional functionality
import os
import glob
from pathlib import Path
import cv2  # For image processing
import tifffile  # For TIFF file handling

# Set style for better visualizations (optional)
plt.style.use('seaborn')
# or
# plt.style.use('default')

# Increase default figure size and DPI for better quality
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['figure.dpi'] = 100

# Set random seed for reproducibility
np.random.seed(42)