
import numpy as np
from cellpose import utils
import numpy as np
import tifffile
from shapely.geometry import Polygon, box, Point
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

def load_centroid_files(input_dir):
    files = glob.glob(os.path.join(input_dir, '*centroids.csv'))
    if not files:
        print(f"No centroid files in {input_dir}")
        return {}
    centroids = {}
    for path in files:
        name = os.path.basename(path)
        try:
            region = int(name.split('_com_')[1].split('_')[0])
        except (IndexError, ValueError):
            print(f"Stopping at unexpected filename: {name}")
            break
        df = pd.read_csv(path)
        centroids[region] = df
        print(f"Region {region}: {len(df)} points")
    return centroids


def load_and_offset_centroids(input_dir, overlapping_squares):
    """
    Load centroid CSV files and add offsets based on polygon bounds.
    """
    centroids_by_region = {}
    
    # Get all centroid CSV files
    files = glob.glob(os.path.join(input_dir, '*centroids.csv'))
    print(f"Found {len(files)} centroid files")
    
    # Debug: print all found files
    print("\nFound files:")
    for f in files:
        print(f"  {os.path.basename(f)}")
    
    for file in files:
        try:
            # Extract region number from filename
            filename = os.path.basename(file)
            # print(f"\nProcessing file: {filename}")
            
            # More robust filename parsing
            parts = filename.split('_')
            # Look for the part after 'com' and before 'centroids'
            com_index = parts.index('com') if 'com' in parts else -1
            if com_index != -1 and com_index + 1 < len(parts):
                region = int(parts[com_index + 1])
                # print(f"Extracted region number: {region}")
            else:
                print(f"Skipping file {filename} - could not extract region number")
                continue
            
            # Get corresponding polygon bounds for this region
            if region <= len(overlapping_squares):
                polygon = overlapping_squares[region - 1]  # region is 1-based
                min_x, min_y, max_x, max_y = map(int, polygon.bounds)
                
                # Load and offset the centroids
                df = pd.read_csv(file)
                # print(f"Loaded {len(df)} points from CSV")
                
                # Add offsets to the centroid coordinates
                df['centroid_x'] = df['centroid_x'] + min_x
                df['centroid_y'] = df['centroid_y'] + min_y
                
                centroids_by_region[region] = df
                print(f"Region {region}: processed {len(df)} points, offset by ({min_x}, {min_y})")
            else:
                print(f"Warning: Region {region} has no corresponding polygon (max region: {len(overlapping_squares)})")
                
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            print(f"File parts: {parts}")
            continue
    
    if not centroids_by_region:
        raise ValueError("No valid centroid files were processed!")
        
    return centroids_by_region

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
    
def map_centroids_to_nonoverlapping(centroids_dict, non_overlapping_squares):
    """
    Maps offset centroids to their corresponding non-overlapping squares.
    Returns both mapped centroids and mapping information for cell IDs.
    """
    mapped_centroids = []
    points_outside = []
    cell_mapping = {}  # To store which cells belong to which non-overlapping region
    
    for region, df in centroids_dict.items():
        # print(f"Processing centroids from region {region}")
        cell_mapping[region] = {}  # Initialize mapping for this region
        
        for idx, row in df.iterrows():
            point = Point(row['centroid_x'], row['centroid_y'])
            point_mapped = False
            cell_id = idx + 1  # Assuming 1-based cell IDs
            
            for square_idx, square in enumerate(non_overlapping_squares):
                if square.contains(point):
                    mapped_data = {
                        'original_region': region,
                        'mapped_region': square_idx + 1,
                        'cell_id': cell_id,
                        'centroid_x': row['centroid_x'],
                        'centroid_y': row['centroid_y'],
                        'cell_type': row.get('cell_type', None),
                        'area': row.get('area', None)
                    }
                    mapped_centroids.append(mapped_data)
                    
                    # Store mapping information
                    if region not in cell_mapping:
                        cell_mapping[region] = {}
                    cell_mapping[region][cell_id] = square_idx + 1
                    
                    point_mapped = True
                    break
            
            if not point_mapped:
                points_outside.append({
                    'original_region': region,
                    'cell_id': cell_id,
                    'centroid_x': row['centroid_x'],
                    'centroid_y': row['centroid_y']
                })
    
    result_df = pd.DataFrame(mapped_centroids)
    outside_df = pd.DataFrame(points_outside)
    
    print(f"\nTotal centroids processed: {len(result_df)}")
    print(f"Points not mapped to any square: {len(outside_df)}")
    if not result_df.empty:
        print("\nCentroids per mapped region:")
        print(result_df['mapped_region'].value_counts().sort_index())
    
    return result_df, outside_df, cell_mapping
    
    # Convert to DataFrame
    result_df = pd.DataFrame(mapped_centroids)
    outside_df = pd.DataFrame(points_outside)
    
    # Print statistics
    print(f"\nTotal centroids processed: {len(result_df)}")
    print(f"Points not mapped to any square: {len(outside_df)}")
    print("\nCentroids per mapped region:")
    if not result_df.empty:
        print(result_df['mapped_region'].value_counts().sort_index())
    
    return result_df, outside_df

def visualize_mapped_centroids(result_df, outside_df, non_overlapping_squares, figsize=(15, 15)):
    """
    Visualizes the mapped centroids and squares.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot non-overlapping squares
    for idx, square in enumerate(non_overlapping_squares):
        x, y = square.exterior.xy
        ax.plot(x, y, 'k-', linewidth=2, alpha=0.7)
        # Add region number label
        centroid = square.centroid
        ax.text(centroid.x, centroid.y, str(idx + 1), 
                horizontalalignment='center', 
                verticalalignment='center',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Plot mapped centroids
    if not result_df.empty:
        scatter = ax.scatter(result_df['centroid_x'], result_df['centroid_y'], 
                           c=result_df['mapped_region'], 
                           cmap='tab20', 
                           alpha=0.6, 
                           s=30,
                           label='Mapped points')
        plt.colorbar(scatter, label='Mapped Region')
    
    # Plot points outside any square in red
    if not outside_df.empty:
        ax.scatter(outside_df['centroid_x'], outside_df['centroid_y'], 
                  color='red', 
                  alpha=0.6, 
                  s=30,
                  label='Unmapped points')
    
    ax.set_title('Mapped Centroids in Non-overlapping Regions')
    ax.legend()
    ax.set_aspect('equal')
    plt.show()



def visualize_mapped_centroids_with_sections(result_df, outside_df, non_overlapping_squares, all_sections, pixelsize, figsize=(15, 15)):
    """
    Visualizes the mapped centroids, squares, and section polygons.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot section polygons first (with alpha)
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, section in enumerate(all_sections):
        # Rescale section coordinates
        x_coords_rescaled = section['x_coords'] / pixelsize
        y_coords_rescaled = section['y_coords'] / pixelsize
        
        color = colors[i % len(colors)]
        
        # Plot section boundary
        ax.plot(x_coords_rescaled, y_coords_rescaled, 
                color=color, linewidth=2, alpha=0.5,
                label=f"{section['name']}")
        
        # Fill section area with alpha
        ax.fill(x_coords_rescaled, y_coords_rescaled, 
                color=color, alpha=0.1)
        
        # Close the polygon line
        if len(x_coords_rescaled) > 2:
            ax.plot([x_coords_rescaled[0], x_coords_rescaled[-1]], 
                    [y_coords_rescaled[0], y_coords_rescaled[-1]], 
                    color=color, linewidth=2, alpha=0.5)
    
    # Plot non-overlapping squares
    for idx, square in enumerate(non_overlapping_squares):
        x, y = square.exterior.xy
        ax.plot(x, y, 'k-', linewidth=1, alpha=0.5)
        # Add region number label
        centroid = square.centroid
        ax.text(centroid.x, centroid.y, str(idx + 1), 
                horizontalalignment='center', 
                verticalalignment='center',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Plot mapped centroids
    if not result_df.empty:
        scatter = ax.scatter(result_df['centroid_x'], result_df['centroid_y'], 
                           c=result_df['mapped_region'], 
                           cmap='tab20', 
                           alpha=0.6, 
                           s=20,
                           label='Mapped points')
        # Make colorbar smaller and horizontal under the plot
        cbar = plt.colorbar(scatter, label='Mapped Region', 
                           orientation='horizontal', 
                           shrink=0.6, 
                           aspect=30,
                           pad=0.1)
    
    ax.set_title('Mapped Centroids with Section Boundaries and Regions')
    # Make the legend font smaller by setting fontsize
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def map_and_filter_outlines(input_dir, overlapping_squares, cell_mapping):
    """
    Load, filter, and map outlines based on the cell mapping from centroids.
    """
    filtered_outlines_list = []
    region_info = []
    
    for region, valid_cells_dict in cell_mapping.items():
        if not valid_cells_dict:  # Skip if no valid cells in this region
            continue
            
        try:
            # Get region bounds
            polygon = overlapping_squares[region - 1]
            min_x, min_y, max_x, max_y = map(int, polygon.bounds)
            
            # Load segmentation data
            # seg_file = os.path.join(input_dir, f'cropped_square_com_{region}_seg.npy')
            seg_file = os.path.join(input_dir, f'cropped_square_com_{region}_seg.npy')
            seg_data = np.load(seg_file, allow_pickle=True).item()
            outlines = utils.outlines_list(seg_data['masks'])
            
            # Filter and map outlines
            for cell_id, mapped_region in valid_cells_dict.items():
                if cell_id <= len(outlines):  # Make sure we have this outline
                    outline = outlines[cell_id - 1]  # Convert to 0-based index
                    # Add offset to outline coordinates
                    mapped_outline = outline + np.array([min_x, min_y])
                    filtered_outlines_list.append(mapped_outline)
                    region_info.append({
                        'original_region': region,
                        'mapped_region': mapped_region,
                        'cell_id': cell_id
                    })
            
            print(f"Region {region}: Processed {len(valid_cells_dict)} outlines")
            
        except Exception as e:
            print(f"Error processing outlines for region {region}: {str(e)}")
            continue
    
    return filtered_outlines_list, pd.DataFrame(region_info)


def visualize_mapped_data(result_df, outside_df, filtered_outlines_list, non_overlapping_squares, figsize=(15, 15)):
    """
    Visualizes the mapped centroids, outlines, and squares.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot non-overlapping squares
    for idx, square in enumerate(non_overlapping_squares):
        x, y = square.exterior.xy
        ax.plot(x, y, 'k-', linewidth=2, alpha=0.7)
        centroid = square.centroid
        ax.text(centroid.x, centroid.y, str(idx + 1), 
                horizontalalignment='center', 
                verticalalignment='center',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Plot mapped centroids
    if not result_df.empty:
        scatter = ax.scatter(result_df['centroid_x'], result_df['centroid_y'], 
                           c=result_df['mapped_region'], 
                           cmap='tab20', 
                           alpha=0.6, 
                           s=30,
                           label='Mapped points')
        plt.colorbar(scatter, label='Mapped Region')
    
    # Plot points outside any square
    if not outside_df.empty:
        ax.scatter(outside_df['centroid_x'], outside_df['centroid_y'], 
                  color='red', 
                  alpha=0.6, 
                  s=30,
                  label='Unmapped points')
    
    # Plot outlines
    for outline in filtered_outlines_list:
        ax.plot(outline[:, 0], outline[:, 1], 'g-', linewidth=1, alpha=0.5)
    
    ax.set_title('Mapped Centroids and Cell Outlines')
    ax.legend()
    ax.set_aspect('equal')
    plt.show()
    
def boundaries_to_table(boundaries):
    # Create an empty list to store data   rows
    data = []
    
    # Loop through each boundary (each cell)
    for cell_id, boundary in enumerate(boundaries, start=1):
        # Loop through each point in the boundary
        for point in boundary:
            # Append cell_id, x, y to the data list
            data.append([cell_id, point[0], point[1]])
    
    # Create a DataFrame with the data
    df = pd.DataFrame(data, columns=['cell_id', 'vertex_x', 'vertex_y'])
    
    return df

def reduce_points(df):
    # Group by cell_id and get counts
    cell_counts = df.groupby('cell_id').size()
    
    new_df = pd.DataFrame()
    
    for cell_id in cell_counts.index:
        cell_data = df[df['cell_id'] == cell_id]
        
        if cell_counts[cell_id] > 150:
            # Keep every other row until we have 50 points
            reduced_data = cell_data.iloc[::2].head(150)
        elif 120 < cell_counts[cell_id] <= 150:
            # Keep every other row until we have 30 points
            reduced_data = cell_data.iloc[::2].head(120)
        else:
            # Keep all points if count <= 30
            reduced_data = cell_data
            
        new_df = pd.concat([new_df, reduced_data])
    
    return new_df.reset_index(drop=True)


def plot_cell_outlines_comparison(df_orig, df_reduced, n=30, rows=5, cols=6):
    # find cell IDs present in both dataframes
    common_ids = np.intersect1d(df_orig['cell_id'].unique(),
                                df_reduced['cell_id'].unique())
    # sample up to n cells
    ids = np.random.choice(common_ids, min(n, len(common_ids)), replace=False)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for ax, cell_id in zip(axes, ids):
        orig = df_orig[df_orig['cell_id'] == cell_id]
        red  = df_reduced[df_reduced['cell_id'] == cell_id]

        # original outline
        ax.plot(orig['vertex_x'], orig['vertex_y'],
                linewidth=2, label='Original', alpha=0.6)
        # reduced outline
        ax.plot(red['vertex_x'],  red['vertex_y'],
                linewidth=2, linestyle='--', label='Reduced', alpha=0.6)

        ax.set_title(f'Cell {cell_id}', fontsize=8)
        ax.invert_yaxis()
        ax.axis('off')
        ax.legend(loc='lower right', fontsize=6)

    # turn off any extra axes
    for ax in axes[len(ids):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def calculate_centroids(df):
    # Group the DataFrame by 'cell_id'
    grouped = df.groupby('cell_id')
    
    # Calculate centroids for each group
    centroids = grouped[['vertex_x', 'vertex_y']].mean().reset_index()
    
    # Rename columns if needed to indicate centroids explicitly
    centroids.rename(columns={'vertex_x': 'centroid_x', 'vertex_y': 'centroid_y'}, inplace=True)
    
    return centroids

def rescale_coordinates(df, scale_factor=0.2125):
    """
    Rescale vertex coordinates by multiplying with scale factor.
    
    Args:
        df: DataFrame with vertex_x and vertex_y columns
        scale_factor: Scaling factor (default: 0.2125)
    
    Returns:
        DataFrame with rescaled coordinates
    """
    df_scaled = df.copy()
    df_scaled['vertex_x'] = df['vertex_x'] * scale_factor
    df_scaled['vertex_y'] = df['vertex_y'] * scale_factor
    return df_scaled
