import pandas as pd
from shapely.geometry import Point, Polygon
from rtree import index

# Filter transcripts to retain only those within the polygon
def is_inside_polygon(row):
    global processed_count
    point = Point(row["x_location"], row["y_location"])
    processed_count += 1
    if processed_count % 10000 == 0:  # Print every 100 rows processed
        print(f"Processed {processed_count} transcripts...")
    return polygon.contains(point)

# Optimized function to find the cell_id for a given x, y location
def find_cell_id_optimized(x, y):
    global processed_count
    point = Point(x, y)
    possible_matches = list(spatial_index.intersection((x, y, x, y)))
    for cell_id in possible_matches:
        if cell_polygons[cell_id].contains(point):
            return cell_id
    return "UNASSIGNED"

# Modified function to handle progress tracking with R-Tree
def process_with_progress(transcripts_df):
    global processed_count
    results = []
    for index, row in transcripts_df.iterrows():
        cell_id = find_cell_id_optimized(row["x_location"], row["y_location"])
        results.append(cell_id)
        processed_count += 1
        if processed_count % 10000 == 0:  # Print every 1000 rows processed
            print(f"Processed {processed_count} transcripts...")
    return results