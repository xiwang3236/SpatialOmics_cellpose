import pandas as pd
from shapely.geometry import Point, Polygon
from rtree import index

spatial_index = index.Index()
cell_polygons = {}

def is_inside_polygon(row, polygon):
    """
    Returns True if the transcript at (x_location, y_location)
    lies inside the given polygon. Also prints progress every 10k rows.
    """
    global processed_count
    processed_count += 1
    if processed_count % 100000 == 0:
        print(f"  iltered {processed_count} rows…")
    pt = Point(row["x_location"], row["y_location"])
    return polygon.contains(pt)

def find_cell_id_optimized(x, y, spatial_index, cell_polygons):
    """
    Given an (x,y) point, uses the R-tree index to find
    which cell polygon contains it. Returns “UNASSIGNED” if none.
    """
    pt = Point(x, y)
    for cid in spatial_index.intersection((x, y, x, y)):
        if cell_polygons[cid].contains(pt):
            return cid
    return "UNASSIGNED"

def assign_cell_ids_with_progress(df, spatial_index, cell_polygons):
    """
    Iterates through df, uses find_cell_id_optimized(), and
    prints progress every 10k rows. Returns a list of cell_ids.
    """
    global processed_count
    processed_count = 0
    cell_ids = []
    for _, row in df.iterrows():
        cid = find_cell_id_optimized(
            row["x_location"], row["y_location"],
            spatial_index, cell_polygons
        )
        cell_ids.append(cid)
        processed_count += 1
        if processed_count % 10000 == 0:
            print(f"  → Assigned {processed_count} rows…")
    return cell_ids