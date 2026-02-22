#!/usr/bin/env python3
"""
georef.py

Lightweight, honest georeferencing of raster map figures using YOLO-discovered
ground control points (GCPs).

Design principles:
- GDAL performs all affine math (translate + warp)
- Only 4 extreme GCPs are used for fitting
- Remaining GCPs (if any) are used for validation
- Projection comparison is performed only when validation is possible
- Emphasis is functional spatial alignment, not survey-grade accuracy
"""

import math
import subprocess
from dataclasses import dataclass
from typing import List, Tuple
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from osgeo import gdal
from pyproj import Transformer
from shapely.geometry import MultiPoint
import math

# Known geographic locations (WGS84) and 
# template.png in deepwater_agentic/geo has the mapping of the names "W1", "W2", ...

LABEL_TO_LONLAT = {
    "W1": (-96.70, 26.32),
    "W2": (-96.30, 28.05),
    "E1": (-83.68, 29.41),
    "E2": (-81.63, 25.29),
    "S1": (-92.95, 25.88),
    "S2": (-87.48, 25.80),
    "S3": (-82.86, 24.63),
    "N1": (-93.27, 29.39),
    "N2": (-88.16, 29.78),
    "CL1": (-96.32, 28.55),
    "CL2": (-94.83, 29.55),
    "CL3": (-88.00, 30.42),
    "CL4": (-82.64, 27.65),
    "CL5": (-80.99, 25.27),
    "CL6": (-84.42, 21.88),
    "CL7": (-87.14, 21.48),
    "CL8": (-91.52, 18.6),
    "CL9": (-97.53, 21.60),
}

# Load image
image_path = "examples/input/OCS-Production-Forecast-2031_figure_page9_0.png"
#image_path = "examples/input/bsee_active_lease.png"

img = cv2.imread(image_path)

# ---------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------

@dataclass
class Label:
    name: str
    px: float
    py: float
    lon: float
    lat: float


# ---------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------

def should_warp(epsg: int) -> bool:
    """
    GDAL warp is only safe for projected CRSs.
    Geographic CRSs (e.g. EPSG:4326) should skip warping.
    """
    return epsg != 4326

def wrap_xy_points(points):
    """
    Convert (name, X, Y) → (name, 0, 0, X, Y)
    """
    return [(name, 0, 0, X, Y) for name, X, Y in points]


def select_extreme_labels(labels: list[Label]) -> list[Label]:
    hull = hull_labels(labels)
    if len(hull) < 3:
        return []

    return select_4_spread(hull)

def transform_lonlat_to_epsg(
    labels: list[Label],
    target_epsg: int
) -> list[Label]:
    """
    Projects lon/lat labels into target EPSG.
    Returns new Label objects with projected x/y stored in lon/lat fields.
    """

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{target_epsg}",
        always_xy=True
    )

    projected = []

    for lbl in labels:
        x, y = transformer.transform(lbl.lon, lbl.lat)
        projected.append(
            Label(
                name=lbl.name,
                px=lbl.px,
                py=lbl.py,
                lon=x,
                lat=y
            )
        )

    return projected

def transform_xy(
    labels: list[Label],
    from_epsg: int,
    to_epsg: int
) -> list[Label]:
    """
    Reprojects Label coordinates from one projected CRS to another.
    Assumes lbl.lon / lbl.lat store X/Y in from_epsg.
    """

    if from_epsg == to_epsg:
        return labels

    transformer = Transformer.from_crs(
        f"EPSG:{from_epsg}",
        f"EPSG:{to_epsg}",
        always_xy=True
    )

    out = []

    for lbl in labels:
        X, Y = transformer.transform(lbl.lon, lbl.lat)
        out.append(
            Label(
                name=lbl.name,
                px=lbl.px,
                py=lbl.py,
                lon=X,
                lat=Y
            )
        )

    return out

def prepare_gcps_for_epsg(
    labels: list[Label],
    epsg: int
) -> list[Label]:
    """
    Returns labels whose lon/lat fields are expressed
    in the target EPSG units (degrees or meters).
    """

    if epsg == 4326:
        return labels  # already lon/lat

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True
    )

    out = []
    for lbl in labels:
        X, Y = transformer.transform(lbl.lon, lbl.lat)
        out.append(
            Label(
                name=lbl.name,
                px=lbl.px,
                py=lbl.py,
                lon=X,
                lat=Y
            )
        )
    return out

def hull_labels(labels: list[Label]) -> list[Label]:
    pts = [(lbl.lon, lbl.lat) for lbl in labels]
    hull = MultiPoint(pts).convex_hull

    if hull.geom_type == "Point":
        return []
    elif hull.geom_type == "LineString":
        coords = list(hull.coords)
    else:  # Polygon
        coords = list(hull.exterior.coords)

    hull_set = set(coords)

    return [
        lbl for lbl in labels
        if (lbl.lon, lbl.lat) in hull_set
    ]

def select_4_spread(labels: list[Label]) -> list[Label]:
    if len(labels) <= 4:
        return labels

    cx = sum(l.lon for l in labels) / len(labels)
    cy = sum(l.lat for l in labels) / len(labels)

    buckets = {0: None, 1: None, 2: None, 3: None}

    for lbl in labels:
        dx = lbl.lon - cx
        dy = lbl.lat - cy
        angle = math.atan2(dy, dx)

        # quadrant index
        q = int((angle + math.pi) // (math.pi / 2)) % 4
        dist = math.hypot(dx, dy)

        if buckets[q] is None or dist > buckets[q][0]:
            buckets[q] = (dist, lbl)

    return [v[1] for v in buckets.values() if v]


def dedup_labels(labels: list[Label]) -> list[Label]:
    by_name = defaultdict(list)
    for lbl in labels:
        by_name[lbl.name].append(lbl)

    deduped = []
    for name, group in by_name.items():
        if len(group) == 1:
            deduped.append(group[0])
        else:
            # simplest possible rule:
            # keep the one closest to the map centroid
            cx = sum(l.px for l in labels) / len(labels)
            cy = sum(l.py for l in labels) / len(labels)

            best = min(
                group,
                key=lambda l: math.hypot(l.px - cx, l.py - cy)
            )
            deduped.append(best)

    return deduped
# ---------------------------------------------------------------------
# GDAL operations
# ---------------------------------------------------------------------

def run_gdal_pipeline(
    image_path: str,
    gcps_lonlat: list,
    epsg: int,
    output_tif: str,
):
    # 1. Attach GCPs
    gcp_tif = output_tif.replace(".tif", "_gcps.tif")

    translate_cmd = [
        "gdal_translate",
        "-of", "GTiff",
        "-a_srs", f"EPSG:{epsg}",
    ]

    for lbl in gcps_lonlat:
        translate_cmd += [
            "-gcp",
            str(lbl.px), str(lbl.py),
            str(lbl.lon), str(lbl.lat),
        ]

    translate_cmd += [image_path, gcp_tif]
    subprocess.run(translate_cmd, check=True)

    # 2. Warp ONLY if projected
    warp_cmd = [
        "gdalwarp",
        "-overwrite",
        "-order", "1",
        "-r", "bilinear",
        "-dstalpha",
        gcp_tif,
        output_tif,
    ]

    # For geographic CRS, explicitly preserve CRS and avoid reprojection
    if epsg == 4326:
        warp_cmd.insert(1, "-t_srs")
        warp_cmd.insert(2, "EPSG:4326")

    subprocess.run(warp_cmd, check=True)

    return output_tif

def extract_warped_points(
    tif_path: str,
    labels: list[Label]
) -> list[Label]:
    """
    Extract projected coordinates of labels from GeoTIFF.
    Returns Labels with lon/lat holding projected X/Y.
    """
    ds = gdal.Open(tif_path)
    gt = ds.GetGeoTransform()

    out = []
    for lbl in labels:
        X = gt[0] + lbl.px * gt[1] + lbl.py * gt[2]
        Y = gt[3] + lbl.px * gt[4] + lbl.py * gt[5]

        out.append(
            Label(
                name=lbl.name,
                px=lbl.px,
                py=lbl.py,
                lon=X,   # projected X
                lat=Y    # projected Y
            )
        )

    return out

from collections import defaultdict

def validate_projection(
    warped_tif: str,
    validation_labels: list[Label],
    warped_epsg: int,
    compare_epsg: int = 5070,
) -> float:
    warped_labels = extract_warped_points(warped_tif, validation_labels)

    warped_proj = transform_xy(warped_labels, warped_epsg, compare_epsg)
    original_proj = transform_lonlat_to_epsg(validation_labels, compare_epsg)

    deviations = [
        math.hypot(w.lon - o.lon, w.lat - o.lat)
        for w, o in zip(warped_proj, original_proj)
    ]

    return max(deviations) / 1000.0

# ---------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------

def georeference_pipeline(
    image_path: str,
    labels: List[Label],
    out_prefix: str
) -> None:
    """
    Main entry point.
    """

    print(f"Detected {len(labels)} control points")

    if len(labels) < 3:
        print("Fewer than 3 control points found. Cannot georeference.")
        return

    if len(labels) == 3:
        print("Only 3 control points found.")
        print("Affine fit underdetermined. Skipping GeoTIFF generation.")
        return

    fit_labels = select_extreme_labels(labels)
    validation_labels = [l for l in labels if l not in fit_labels]

    # -----------------------------------------------------------------
    # Case A: exactly 4 points → WGS only
    # -----------------------------------------------------------------
    if not validation_labels:
        print("Exactly 4 control points found.")
        print("No independent validation possible.")
        print("Generating WGS84 GeoTIFF only.")

        out_tif = f"{out_prefix}_wgs84.tif"
        run_gdal_pipeline(image_path, fit_labels, 4326, out_tif)
        print(f"Output written to {out_tif}")
        return

    # -----------------------------------------------------------------
    # Case B: validation available → compare projections
    # -----------------------------------------------------------------
    candidates = [4326, 5070]
    results = {}

    for epsg in candidates:
        out_tif = f"{out_prefix}_{epsg}.tif"
        print(f"\nRunning GDAL for EPSG:{epsg}")

        gcps_for_gdal = prepare_gcps_for_epsg(fit_labels, epsg)

        run_gdal_pipeline(image_path=image_path, 
                          gcps_lonlat = gcps_for_gdal, 
                          epsg = epsg, 
                          output_tif = out_tif)

        err = validate_projection(out_tif, validation_labels, epsg, compare_epsg = 5070)

        results[epsg] = (err, out_tif)
        print(f"Validation max deviation: {err / 1000:.2f} km")

    best_epsg = min(results, key=lambda k: results[k][0])
    best_err, best_tif = results[best_epsg]

    print("\n==============================")
    print(f"Selected CRS: EPSG:{best_epsg}")
    print(f"Max deviation: {best_err / 1000:.2f} km")
    print(f"Final GeoTIFF: {best_tif}")
    print("==============================")

# Output the table of coordinates
import csv

# Load trained YOLOv8 segmentation model
model = YOLO("models/best.pt")

# Run inference
results = model(image_path, conf=0.25)

# Copy image for drawing
vis = img.copy()

for r in results:
    if r.masks is None:
        continue

    # Loop over detected masks
    class_ids = r.boxes.cls.cpu().numpy().astype(int)

    detections =[]

    for i, (polygon, cls_id) in enumerate(zip(r.masks.xy, class_ids)):
        pts = np.array(polygon, dtype=np.int32)

        # Draw filled mask (semi-transparent)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        alpha = 0.4
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

        # Compute centroid (area-weighted)
        M = cv2.moments(pts)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            label = model.names[cls_id]

            # Draw centroid
            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(
                vis,
                label,
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
            detections.append({"label": label, "cx": cx, "cy": cy})
# Display final result
cv2.imshow("YOLO Masks + Centroids", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Output the table of coordinates
import csv

gcp_entries = []

with open("gcps.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "pixel_x", "pixel_y", "lon", "lat"])

    for det in detections:
        label = det["label"]
        if label not in LABEL_TO_LONLAT:
            print(label)
            continue  # skip labels without known geo location

        lon, lat = LABEL_TO_LONLAT[label]
        px, py = det["cx"], det["cy"]

        writer.writerow([label, px, py, lon, lat])
        gcp_entries.append((label, px, py, lon, lat))

labels = [Label(*t) for t in gcp_entries]
labels = dedup_labels(labels)

georeference_pipeline(
    image_path,
    labels,
    out_prefix="examples/output/map_georef"
)


