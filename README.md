# **Re-activating Spatial Intelligence in Static Map Imagery**

This repository contains an automated pipeline that uses modern machine learning and computer vision to georeference static map figures — producing spatially enabled GeoTIFFs from images that originally lack geospatial metadata. The workflow leverages YOLO for control point extraction, projection hypothesis testing, geometric validation, and GDAL warping. The result supports spatial analytical systems such as **GeoRAG** (Geospatial Retrieval Augmented Generation).

---

## **Abstract**

Many valuable geographic insights are trapped inside PNG, JPG, or PDF map figures, but these files typically lack the coordinate reference system (CRS) or geospatial metadata that modern GIS tools require. This project demonstrates how machine learning and geometric reasoning can extract latent spatial signals and reintroduce them into geospatial workflows, enabling map imagery to participate in spatial analytics and GeoRAG applications.

Rather than focusing on survey-grade geodetic precision, the pipeline prioritizes **functional spatial alignment** sufficient for analytical integration at regional scales. For regulatory, engineering, or cadastral usage, authoritative data sources should always be used.

---

## **How It Started**

This work builds on the Medium article titled:

[*“Automatic GeoRAG: Grounding Vision Models into GIS with YOLO, GeoTIFFs, and Agentic LLMs”*](https://medium.com/@stpiwaco/automatic-georag-grounding-vision-models-into-gis-with-yolo-geotiffs-and-agentic-llms-ee6fcacaf134)

The article described a prototype pipeline that uses object detection and classic GIS georeferencing to recover spatial context from map figures with specific features (like protraction grids typical in offshore oil & gas).

This repository expands and hardens that concept by:

* Extending YOLO label classes beyond protraction to include coastline and other geo-features

* Adding automated projection hypothesis testing

* Incorporating geometric consistency scoring

* Providing a production-ready GDAL georeferencing pipeline

---

## **Pipeline Overview**

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

## **Repository Structure**

geo-rag-map-reactivation/  
├── README.md  
├── LICENSE  
├── requirements.txt  
├── models/  
│   └── README.md   \# download YOLO weights  
├── src/  
│   ├── detection.py  
│   ├── projection\_selection.py  
│   ├── georeference.py  
│   └── main.py  
├── examples/  
│   ├── input/  
│   └── output/  
├── docs/  
│   └── images/

---

## **YOLO Training**

Label Studio was utilized to train a YOLO11 segmentation model on 85 randomized Gulf of Mexico maps. The dataset includes varied imagery from BOEM and BSEE, spanning multiple coordinate reference systems and scales. I labeled 19 distinct features (9 coastlines; 10 protraction grids), accelerating the workflow via Label Studio’s automated outlining backend. The model was trained using `yolo11n-seg.pt` for 150 epochs, reaching optimal convergence at epoch 130\.

---

## **Reproducibility**

The repository includes:

* End-to-end pipeline code

* YOLO model weights (see Releases section)

* Example input figures

* Sample output GeoTIFFs

To reproduce results:

`pip install -r requirements.txt`  
`python src/main.py --input example.png --output output.tif`

---

## **Collaboration**

This project is part of ongoing exploration in GeoAI, spatial reasoning, and GeoRAG systems.  
If your organization is exploring similar directions, collaboration is welcome.

You can connect with me on LinkedIn:  
[David W](http://www.linkedin.com/in/david-w-techie)
