# Re-activating Spatial Intelligence in Static Map Imagery

> Re-activating spatial intelligence locked inside static map imagery.

This repository contains an automated pipeline that uses modern machine learning and computer vision to georeference static map figures — producing spatially enabled GeoTIFFs from images that originally lack geospatial metadata. The workflow leverages YOLO for control point extraction, projection hypothesis testing, geometric validation, and GDAL warping. The result supports spatial analytical systems such as **GeoRAG** (Geospatial Retrieval Augmented Generation).

---

## Project Vision

Many valuable geographic insights are trapped inside PNG, JPG, or PDF map figures, but these files typically lack the coordinate reference system (CRS) or geospatial metadata that modern GIS tools require. This project demonstrates how machine learning and geometric reasoning can extract latent spatial signals and reintroduce them into geospatial workflows, enabling map imagery to participate in spatial analytics and GeoRAG applications.

Rather than focusing on survey-grade geodetic precision, the pipeline prioritizes **functional spatial alignment** sufficient for analytical integration at regional scales. For regulatory, engineering, or cadastral usage, authoritative data sources should always be used.

---

## How It Started

This work builds on the Medium article titled:

> *“Automatic GeoRAG: Grounding Vision Models into GIS with YOLO, GeoTIFFs, and Agentic LLMs”*

The article described a prototype pipeline that uses object detection and classic GIS georeferencing to recover spatial context from map figures with specific features (like protraction grids typical in offshore oil & gas).  

This repository expands and hardens that concept by:

- Extending YOLO label classes beyond protraction to include coastline and other geo-features
- Adding automated projection hypothesis testing
- Incorporating geometric consistency scoring
- Providing a production-ready GDAL georeferencing pipeline

---

## Pipeline Overview

The high-level workflow is:

1. **Detection**
   - YOLO identifies labeled map features (e.g., coastline points, grid nodes).
2. **Projection Hypothesis Testing**
   - Evaluate candidate CRSs (e.g., EPSG:4326, EPSG:5070, UTM zones) with geometric residual scoring.
3. **Triplet-Based Consistency**
   - Test combinations of three control points to avoid noisy/mislabeled points.
4. **Spatial Spread Selection**
   - Select 4 extreme control points with maximal spatial spread.
5. **GDAL Georeferencing**
   - Attach GCPs via `gdal_translate`, warp with `gdalwarp` (affine).
6. **Output**
   - Aligned GeoTIFF suitable for GIS indexing and GeoRAG systems.

---

## Repository Structure

```plaintext
geo-rag-map-reactivation/
├── README.md
├── LICENSE
├── requirements.txt
├── models/
│   └── README.md   # download YOLO weights
├── src/
│   ├── detection.py
│   ├── projection_selection.py
│   ├── georeference.py
│   └── main.py
├── examples/
│   ├── input/
│   └── output/
├── docs/
│   └── images/
