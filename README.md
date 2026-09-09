# Recreation Ecosystem Services Module

`GEP_recreation` — preliminary analysis of GEP recreation

**Part of the Global Gross Ecosystem Product (GEP) Project**  
Natural Capital Project's Earth Economy Modelers (NatCap TEEMs)  
University of Minnesota

[NatCap TEEMs Project Website](https://natcapteems.umn.edu)

## Overview

This repository contains the recreation ecosystem services module for the Global Gross Ecosystem Product project. The module estimates the economic value of nature-based recreation worldwide by modeling how people interact with and benefit from natural environments.

**Status:** Research in progress

## What This Module Does

The recreation module quantifies two types of nature-based recreation:

- **Recreation from Home**: Visits to natural areas by local residents traveling from their residential locations
- **Recreation from Lodging**: Visits by tourists staying at accommodation facilities (hotels, lodging)

The approach combines geospatial data on natural environments (forests, protected areas, water bodies, grasslands) with accessibility infrastructure (roads, urban proximity) to identify high-quality recreation sites globally. Population (or overnight-stay) demand is then allocated to these sites using a distance-banded, conserved allocation model, and priced at each traveler's realized round-trip cost.

## Geospatial Methods

This module employs several GIS techniques:

### Spatial Analysis

- **Multi-criteria site classification**: Combines environmental quality indices (land cover, protected area status) with accessibility indices (urban proximity, road distance) using matrix-based classification
- **Site dissolution**: Contiguous high-quality-site pixels are merged into discrete sites via connected-component labeling, so a single large park is treated as one destination rather than as many independent site pixels
- **Distance-banded allocation**: Each population pixel's per-capita trip budget is split into non-overlapping distance bands (rings). Within each band, the fixed budget allocated to that band is divided evenly across whatever sites fall within it — never re-drawn per site — so a population pixel near several sites in the same distance range doesn't generate more total trips than an otherwise-identical pixel near just one
- **Per-group dynamic travel cutoff**: The maximum distance a group of travelers is modeled as willing to go scales with that group's own fitted distance-decay rate, rather than one fixed radius applied uniformly worldwide
- **Unmet demand**: Trip budget that cannot be realized — either because no site falls within a population pixel's reachable distance bands, or because it lies beyond that group's cutoff — is tracked explicitly as its own output, rather than silently discarded

### Demand Model

- **Closed-form exponential distance decay**: Trip-distance behavior is represented by an exponential density, calibrated per parameter group by two coefficients — a participation level (per-capita annual visit rate at zero distance) and a distance-decay rate. Band-level allocation shares are computed from this density's closed-form CDF, avoiding numerical integration
- **Parameter groups**: Countries are mapped to demand parameters via a `param_group` key, allowing coverage at country, regional, or archetype resolution as calibration data allows, without changing the underlying model
- **Travel cost valuation**: Economic value is calculated on a capital-account (price × quantity) basis — realized visits multiplied by each traveler's round-trip travel cost, using country-specific fuel prices at each traveler's *origin*, not a consumer-surplus / demand-curve valuation

### Global Raster Processing

- **Multi-resolution data integration**: Processes global datasets at 1km resolution
- **Streaming, windowed computation**: Rather than loading full global rasters into memory, the flow model streams a single pass over the parameter-group raster to find each group's spatial extent, then processes each group as a clipped, buffered window read directly from disk — keeping peak memory bounded by one group's extent rather than global raster size
- **Grid-aware banding**: Distance bands are sized relative to the raster's pixel resolution, so no group is subdivided into bands narrower than the grid can resolve
- **Zonal statistics and aggregation**: Summarizes recreation values by country using ISO3 administrative boundaries, aggregated from site-level results

### Tourism Allocation

- **Hotel point density mapping**: Rasterizes global hotel location data to allocate tourist overnight stays
- **Country-level overnight distribution**: Integrates UN World Tourism Organization (UNWTO) data with spatial hotel patterns to map tourist recreation potential

## Framework Alignment

This work follows the System of Environmental-Economic Accounting – Ecosystem Accounting (SEEA EA) framework, treating recreation as a final non-material ecosystem service that links ecological conditions directly to human well-being.

## Technical Stack

**Core Libraries:**

- **hazelbean**: Project workflow management and hierarchical task execution
- **pygeoprocessing**: High-performance raster processing and spatial operations
- **GDAL/OGR**: Geospatial data I/O and transformations
- **geopandas**: Vector data handling and spatial joins
- **pandas/numpy**: Data manipulation and array operations
- **scipy.ndimage**: Connected-component labeling for site dissolution

**Key Functions:**

- Raster calculator operations for multi-band analysis
- Ring-kernel convolution for distance-banded allocation
- Vector-to-raster conversion with attribute burning
- Zonal aggregation across administrative boundaries


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
3. **Recreation Site Ranking**: Combines environment and accessibility indices to identify high-quality recreation sites (rank 1–9)
4. **Site Dissolution**: Merges contiguous high-quality-site pixels into discrete sites via connected-component labeling
5. **Overnight Allocation**: Distributes UNWTO country-level tourist overnight data to hotel locations
6. **Demand Parameter Loading**: Loads a parameter table (`iso3_r250_id`, `param_group`, participation and distance-decay coefficients) and builds the corresponding spatial parameter-group raster
7. **Flow Calculation**: For each parameter group, computes a dynamic distance cutoff and a set of distance bands from that group's decay rate, then allocates each population pixel's trip budget across reachable sites within those bands — conserving total trips and tracking any unrealized (unmet) demand
8. **Economic Valuation**: Prices realized visits at each traveler's round-trip travel cost, using origin-country fuel prices
9. **Country Aggregation**: Summarizes visits, value, and unmet demand by ISO3 country code for policy analysis

---

## Model Outputs

For both the resident ("daily") and tourist flow calculations, the pipeline produces:

- **Site-level visits and value**: aggregated per dissolved site, with the site's dominant country attributed
- **Unmet demand**: population-side trip budget that could not be allocated to any reachable site, either within-band or beyond a group's cutoff
- **Parameter-group QA table**: per-group budget, realized, and unmet totals, plus the resolved cutoff distance and band count — useful for sanity-checking coverage and calibration before trusting downstream results
- **Country-level aggregates**: visits and value summed by ISO3 code, combining resident and tourist flows

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
