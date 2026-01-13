# Recreation Ecosystem Services Module

**Part of the Global Gross Ecosystem Product (GEP) Project**  
Natural Capital Project's Earth Economy Modelers (NatCap TEEMs)  
University of Minnesota

[NatCap TEEMs Project Website](https://natcapteems.umn.edu)

## Overview

This repository contains the recreation ecosystem services module for the Global Gross Ecosystem Product project. The module estimates the economic value of nature-based recreation worldwide by modeling how people interact with and benefit from natural environments.

**Status:** Research in progress

---

## What This Module Does

The recreation module quantifies two types of nature-based recreation:

- **Recreation from Home**: Visits to natural areas by local residents traveling from their residential locations
- **Recreation from Lodging**: Visits by tourists staying at accommodation facilities (hotels, lodging)

The approach combines geospatial data on natural environments (forests, protected areas, water bodies, grasslands) with accessibility infrastructure (roads, urban proximity) to identify high-quality recreation sites globally. Economic values are derived from gravity-based visit models and travel cost calculations based on distance and country-specific fuel costs.

---

## Geospatial Methods

This module employs several GIS techniques:

### Spatial Analysis

- **Multi-criteria site classification**: Combines environmental quality indices (land cover, protected area status) with accessibility indices (urban proximity, road distance) using matrix-based classification
- **Kernel-based spatial convolution**: Uses distance-weighted kernels to model recreation visits flowing from population centers to recreation sites at multiple distance buffers (1-4 km zones)
- **Gravity model implementation**: Applies logistic gravity models calibrated to population density to estimate visit frequencies
- **Travel cost valuation**: Calculates economic value using distance-weighted travel costs with country-specific fuel price data

### Global Raster Processing

- **Multi-resolution data integration**: Processes global datasets at 1km resolution (~1 million sq km pixels)
- **Efficient memory management**: Uses `pygeoprocessing.raster_calculator` for streaming computation on large global rasters
- **Zonal statistics and aggregation**: Summarizes recreation values by country using ISO3 administrative boundaries

### Tourism Allocation

- **Hotel point density mapping**: Rasterizes global hotel location data to allocate tourist overnight stays
- **Country-level overnight distribution**: Integrates UN World Tourism Organization (UNWTO) data with spatial hotel patterns to map tourist recreation potential

---

## Framework Alignment

This work follows the System of Environmental-Economic Accounting – Ecosystem Accounting (SEEA EA) framework, treating recreation as a final non-material ecosystem service that links ecological conditions directly to human well-being.

---

## Technical Stack

**Core Libraries:**

- **hazelbean**: Project workflow management and hierarchical task execution
- **pygeoprocessing**: High-performance raster processing and spatial operations
- **GDAL/OGR**: Geospatial data I/O and transformations
- **geopandas**: Vector data handling and spatial joins
- **pandas/numpy**: Data manipulation and array operations

**Key Functions:**

- Raster calculator operations for multi-band analysis
- Kernel convolution for spatial proximity modeling
- Vector-to-raster conversion with attribute burning
- Zonal aggregation across administrative boundaries

---

## Repository Structure

```
recreation/
├── run_recreation.py              # Main execution script with task tree
├── recreation_tasks.py            # Core processing functions and workflow tasks
├── README.md                      # This file
├── .gitignore                     
├── recreation_methods/            # Documentation and methods
│   ├── recreation_methods.qmd     # Quarto source document
│   ├── recreation_methods.html    # Rendered methods documentation
│   ├── recreation_methods.docx    # Word version
│   └── assets/                    # Figures and diagrams
└── [project output directories created during runtime]
```

---

## Key Processing Steps

1. **Environment Index Generation**: Classifies pixels by natural environment quality using weighted land cover and protected area data
2. **Accessibility Index Generation**: Classifies pixels by human accessibility using urban proximity and road distance
3. **Recreation Site Ranking**: Combines environment and accessibility indices to identify high-quality recreation sites (rank 1-9)
4. **Overnight Allocation**: Distributes UNWTO country-level tourist overnight data to hotel locations
5. **Visit Calculation**: Uses distance-weighted gravity models to estimate visits flowing from population/lodging to recreation sites
6. **Economic Valuation**: Calculates travel costs based on distance and country-specific fuel prices
7. **Country Aggregation**: Summarizes results by ISO3 country codes for policy analysis

---

## Contact

Matthew Braaksma  
PhD Student, Department of Applied Economics  
University of Minnesota  
braak014@umn.edu

## Authors

Matthew Braaksma, University of Minnesota

Spencer Wood, University of Washington

Lifeng Ren, University of Minnesota

Ryan McWay, University of Minnesota

---

*This is ongoing research. Methods and results are subject to revision.*