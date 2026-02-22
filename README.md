# **Re-activating Spatial Intelligence in Static Map Imagery**

## **Abstract**

Many valuable geographic insights are trapped inside PNG, JPG, or PDF map figures, but these files typically lack the coordinate reference system (CRS) or geospatial metadata that modern GIS tools require. This project demonstrates how machine learning and geometric reasoning can extract latent spatial signals and reintroduce them into geospatial workflows, enabling map imagery to participate in spatial analytics and GeoRAG applications.

Rather than focusing on survey-grade geodetic precision, the pipeline prioritizes **functional spatial alignment** sufficient for analytical integration at regional scales. For regulatory, engineering, or cadastral usage, authoritative data sources should always be used.

This work builds on the Medium article titled:

[*“Automatic GeoRAG: Grounding Vision Models into GIS with YOLO, GeoTIFFs, and Agentic LLMs”*](https://medium.com/@stpiwaco/automatic-georag-grounding-vision-models-into-gis-with-yolo-geotiffs-and-agentic-llms-ee6fcacaf134)

The article described a prototype pipeline that uses object detection and classic GIS georeferencing to recover spatial context from map figures with specific features (like protraction grids typical in offshore oil & gas).

This repository expands and hardens that concept by:

* Extending YOLO label classes beyond protraction to include coastline and other geo-features

* Adding automated projection hypothesis testing

* Incorporating geometric consistency scoring

* Providing a production-ready GDAL georeferencing pipeline

---

## **Projection Strategy**

The pipeline currently evaluates two coordinate reference systems:

* **EPSG:4326 (WGS84)** — geographic (lon/lat)

* **EPSG:5070 (NAD83 / CONUS Albers)** — projected, equal-area

Both are derived using **GDAL affine georeferencing** from automatically detected control points.

### **Why these two?**

* EPSG:4326 (WGS84) is the most widely used geographic coordinate system in the world, serving as the standard for everything from GPS navigation and satellite surveying to global web data exchange.

* While UTM Zone 15 and 16 offer high local accuracy, the Gulf of Mexico spans multiple zones, leading to significant distortion at the fringes of either system.  Consequently, EPSG:5070 (NAD83 / Albers) was selected to provide physically meaningful area measurements and consistent shape preservation across the entire study area.

---

## **YOLO Training**

Label Studio was utilized to train a YOLO11 segmentation model on 85 randomized Gulf of Mexico maps. The dataset includes varied imagery from BOEM and BSEE, spanning multiple coordinate reference systems and scales. I labeled 19 distinct features (9 coastlines; 10 protraction grids), accelerating the workflow via Label Studio’s automated outlining backend. The model was trained using `yolo11n-seg.pt` for 150 epochs, reaching optimal convergence at epoch 130\.

A map being labeled with some of the 19 features in Label Studio is shown below:

<img width="1684" height="1017" alt="image" src="https://github.com/user-attachments/assets/b863bb7d-b0a8-4dc3-8d6b-62f94f0dfc71" />

---

## **How Projection Selection Works**

1. The centroid of each YOLO discovered area forms a **GCP** (**Ground Control Point**).

2. A set of 4 **maximum spreading control points** is selected to anchor the affine transformation.

3. GDAL is used to generate GeoTIFFs for each candidate CRS.

4. Remaining detected labels (not used in fitting) are used for validation.

5. The **maximum spatial deviation** is computed in projected meters.

---

## **Reproducibility**

The repository includes:

* End-to-end pipeline code

* YOLO model weights (see Releases section)

* Example input figures

* Sample output GeoTIFFs

To reproduce results:

`pip install -r requirements.txt`  
`python src/main.py`

The Python script will show the labeled segmentation areas YOLO discovered and their respective centroids in a map window as shown here:

<img width="1615" height="996" alt="Screenshot 2026-02-22 061940" src="https://github.com/user-attachments/assets/b3aa3a80-56ea-47cf-bc69-3f048c3bdb72" />

The spatially enabled files can be brought in QGIS as other shapefile and SDE layers:

<img width="1913" height="999" alt="Screenshot 2026-02-22 062701" src="https://github.com/user-attachments/assets/19e7ad75-6780-48da-b70f-b91461291224" />
---

## **A Note to the Geospatial Community**

This work is offered with respect to the rigor of geodesy and cartography.

It is not a replacement for formal georeferencing workflows, but rather an exploration of how **vision models, GDAL, and modern Python tooling** can help recover spatial context within **100 meters** accuracy from otherwise inert imagery at scale.

If it helps even a fraction of legacy maps participate in modern geospatial analysis and retrieval systems, it has done its job.

---

## **Collaboration**

This project is part of ongoing exploration in GeoAI, spatial reasoning, and GeoRAG systems.  
If your organization is exploring similar directions, collaboration is welcome.

You can connect with me on LinkedIn:  
[David W](http://www.linkedin.com/in/david-w-techie)
