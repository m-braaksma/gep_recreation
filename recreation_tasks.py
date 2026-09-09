"""
recreation_tasks.py  (v2, consolidated)

CHANGES FROM v1:
  - recreation_site_analysis now dissolves contiguous HQ pixels into
    discrete site blobs (merge_hq_sites) as its final step, so downstream
    reporting is per real-world site, not per pixel.
  - The old per-site-pixel convolution + per-country linear-demand-curve
    value model is replaced by a single param-table-driven flow model:
    one continuous equation f(d) = participation_param * exp(-distance_param*d)
    per param_group, evaluated via two convolution passes (visits, value).
    See calculate_recreation_flows for the full derivation.
  - calculate_daily_recreation_into_sites_kernel / _tourist_ keep their
    original names (task-tree compatible) but now call
    calculate_recreation_flows under the hood.
  - aggregate_results_iso3 now reads the site-level CSVs the flow
    calculation writes directly, rather than summing per-pixel rasters.

PARAMETER TABLE (external input, replaces on-the-fly survey fitting):
  Expected columns: iso3_r250_id, unwto_name, param_group,
                     participation_param, distance_param
  - param_group: arbitrary integer, purely a computational grouping key
    (country-level = give every country its own value; continent-level =
    share a value across several countries). Doesn't need to mean anything
    outside this pipeline.
  - participation_param (a): per-capita annual visit rate at distance 0.
    Governs the TOTAL level of realized visits and unmet demand.
  - distance_param (b): exponential decay rate (per km). Governs the
    SPATIAL SPLIT between nearby sites (participation_param cancels out of
    the normalized allocation share, so this parameter alone shapes "how
    far people are willing to go" independent of "how often they go").
  All countries sharing a param_group must have identical
  participation_param/distance_param (that's what makes them one group);
  load_recreation_params validates this.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import pygeoprocessing
from osgeo import gdal, osr
from scipy import ndimage


# ===========================================================================
# Environment / accessibility / site ranking (UNCHANGED from v1)
# ===========================================================================

def generate_environment_index(p):
    inputs = {
        'cropland': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/lulc/lulc_esa_seals7_2010_1km_cropland_share.tif'),
        'forest': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/lulc/lulc_esa_seals7_2010_1km_forest_share.tif'),
        'grassland': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/lulc/lulc_esa_seals7_2010_1km_grassland_share.tif'),
        'othernat': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/lulc/lulc_esa_seals7_2010_1km_othernat_share.tif'),
        'urban': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/lulc/lulc_esa_seals7_2010_1km_urban_share.tif'),
        'water': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/lulc/lulc_esa_seals7_2010_1km_water_share.tif'),
        'pa': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/pa/wdpa_accessible_1km_share.tif'),
    }
    output_path = os.path.join(p.cur_dir, 'env_index_1km.tif')
    if os.path.exists(output_path):
        p.L.info(f"Environment index already exists: {output_path}")
        return

    lulc_scores = {'cropland': 0.4, 'forest': 1.0, 'grassland': 0.7, 'othernat': 0.85,
                   'water': 1.0, 'urban': 0.05, 'other': 0.0}
    pa_score = 1.0
    env_classify = {'high': 1.5, 'med_high': 1.0, 'med_low': 0.5, 'low': 0}
    nodata_value = -1

    def environment_index_op(crop, forest, grass, othernat, urban, water, pa):
        lulc_potential = (
            lulc_scores['cropland'] * crop + lulc_scores['forest'] * forest +
            lulc_scores['grassland'] * grass + lulc_scores['othernat'] * othernat +
            lulc_scores['urban'] * urban + lulc_scores['water'] * water)
        combined = lulc_potential + pa_score * pa
        return np.select(
            condlist=[combined > env_classify['high'], combined > env_classify['med_high'],
                      combined > env_classify['med_low'], combined >= env_classify['low']],
            choicelist=[3, 2, 1, 0], default=0)

    missing_files = [f"{k}: {v}" for k, v in inputs.items() if not os.path.exists(v)]
    if missing_files:
        p.L.warning("Missing input files for generate_environment_index:\n" + "\n".join(missing_files))
        raise FileNotFoundError("Missing input files for generate_environment_index:\n" + "\n".join(missing_files))

    base_list = [(inputs[k], 1) for k in ['cropland', 'forest', 'grassland', 'othernat', 'urban', 'water', 'pa']]
    p.L.info(f"Calculating environment index and saving to: {output_path}")
    pygeoprocessing.raster_calculator(base_list, environment_index_op, output_path,
                                       gdal.GDT_Int32, nodata_value, calc_raster_stats=True)
    p.L.info(f"Environment index calculation completed: {output_path}")


def generate_accessibility_index(p):
    inputs = {
        'urban': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/lulc/lulc_esa_seals7_2010_1km_urban_share.tif'),
        'road': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/distance_to_roads/distance_to_road_1km_corrected.tif')
    }
    output_path = os.path.join(p.cur_dir, 'accessibility_index_1km.tif')
    if os.path.exists(output_path):
        p.L.info(f"Accessibility index already exists: {output_path}")
        return

    urban_classify = {'high': 0.9, 'med_high': 0.5, 'med': 0.1, 'med_low': 0.001, 'low': 0}
    road_classify = {'high': 50, 'med_high': 20, 'med': 5, 'med_low': 2, 'low': 0}
    urban_road_matrix = np.array([
        [1, 1, 2, 3, 4], [1, 1, 2, 3, 4], [2, 2, 2, 4, 5],
        [3, 3, 4, 5, 5], [3, 4, 4, 5, 5]])
    nodata_value = -1

    def accessibility_op(urban, roads):
        urban_class = np.select(
            condlist=[urban > urban_classify['high'], urban > urban_classify['med_high'],
                      urban > urban_classify['med'], urban > urban_classify['med_low'],
                      urban >= urban_classify['low']],
            choicelist=[4, 3, 2, 1, 0], default=0)
        road_class = np.select(
            condlist=[roads > road_classify['high'], roads > road_classify['med_high'],
                      roads > road_classify['med'], roads > road_classify['med_low'],
                      roads >= road_classify['low']],
            choicelist=[0, 1, 2, 3, 4], default=0)
        return urban_road_matrix[urban_class, road_class].astype(np.int32)

    missing_files = [f"{k}: {v}" for k, v in inputs.items() if not os.path.exists(v)]
    if missing_files:
        p.L.warning("Missing input files for generate_accessibility_index:\n" + "\n".join(missing_files))
        raise FileNotFoundError("Missing input files for generate_accessibility_index:\n" + "\n".join(missing_files))

    base_list = [(inputs['urban'], 1), (inputs['road'], 1)]
    p.L.info(f"Calculating accessibility index and saving to: {output_path}")
    pygeoprocessing.raster_calculator(base_list, accessibility_op, output_path, gdal.GDT_Int32, nodata_value)
    p.L.info(f"Accessibility index calculation completed: {output_path}")


def rank_rec_sites(p):
    inputs = {
        'env_index': os.path.join(p.generate_environment_index_dir, 'env_index_1km.tif'),
        'accessibility_index': os.path.join(p.generate_accessibility_index_dir, 'accessibility_index_1km.tif')
    }
    output_path = os.path.join(p.cur_dir, 'rec_sites_ranked_1km.tif')
    if os.path.exists(output_path):
        p.L.info(f"Recreation site ranking already exists: {output_path}")
        return

    site_matrix = np.array([
        [1, 1, 4, 7], [1, 4, 4, 7], [2, 2, 8, 8], [3, 5, 5, 9], [3, 6, 6, 9]])
    nodata_value = -1

    def site_op(acc_class, env_class):
        acc_class = acc_class - 1
        return site_matrix[acc_class, env_class].astype(np.int16)

    missing_files = [f"{k}: {v}" for k, v in inputs.items() if not os.path.exists(v)]
    if missing_files:
        p.L.warning("Missing input files for rank_rec_sites:\n" + "\n".join(missing_files))
        raise FileNotFoundError("Missing input files for rank_rec_sites:\n" + "\n".join(missing_files))

    base_list = [(inputs['accessibility_index'], 1), (inputs['env_index'], 1)]
    p.L.info(f"Calculating site classification and saving to: {output_path}")
    pygeoprocessing.raster_calculator(base_list, site_op, output_path, gdal.GDT_Int32, nodata_value)
    p.L.info(f"Recreation site classification completed: {output_path}")


def find_high_quality_rec_sites(p):
    inputs = {'ranked_sites': os.path.join(p.recreation_site_analysis_dir, 'rec_sites_ranked_1km.tif')}
    output_path = os.path.join(p.cur_dir, 'high_quality_sites_1km.tif')
    if os.path.exists(output_path):
        p.L.info(f"High quality sites already identified: {output_path}")
        return

    def hq_op(site_class):
        return np.where(site_class == 9, 1, 0).astype(np.int32)

    if not os.path.exists(inputs['ranked_sites']):
        p.L.warning(f"Missing input file for find_high_quality_rec_sites: {inputs['ranked_sites']}")
        raise FileNotFoundError(f"Missing input file for find_high_quality_rec_sites: {inputs['ranked_sites']}")

    p.L.info(f"Identifying high quality recreation sites and saving to: {output_path}")
    pygeoprocessing.raster_calculator([(inputs['ranked_sites'], 1)], hq_op, output_path, gdal.GDT_Int32, -1)
    p.L.info(f"High quality sites identification completed: {output_path}")


def merge_hq_sites(p):
    """
    Dissolve contiguous high-quality-site pixels into discrete site blobs
    (8-connected). Fixes the per-pixel double-counting the original
    per-site-pixel convolution approach had: a 65-pixel park becomes 1
    site, not 65 independent destinations.

    Output: hq_sites_dissolved_1km.tif (0 = not a site, 1..N = site id)
    Also writes hq_site_pixel_counts.csv -- a quick fragmentation check
    worth glancing at before running the full flow calculation (a few
    thousand large blobs vs hundreds of thousands of tiny ones changes how
    much the per-param-group buffered convolution costs).
    """
    input_path = os.path.join(p.recreation_site_analysis_dir, 'high_quality_sites_1km.tif')
    output_path = os.path.join(p.cur_dir, 'hq_sites_dissolved_1km.tif')
    site_sizes_path = os.path.join(p.cur_dir, 'hq_site_pixel_counts.csv')

    if os.path.exists(output_path):
        p.L.info(f"Dissolved HQ sites already exist: {output_path}")
        return
    if not os.path.exists(input_path):
        p.L.warning(f"Missing input file for merge_hq_sites: {input_path}")
        raise FileNotFoundError(f"Missing input file for merge_hq_sites: {input_path}")

    p.L.info("Dissolving contiguous high-quality-site pixels into discrete sites")
    hq_nodata = pygeoprocessing.get_raster_info(input_path)['nodata'][0]
    ds = gdal.Open(input_path)
    hq_arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    is_hq = (hq_arr == 1) if hq_nodata is None else ((hq_arr != hq_nodata) & (hq_arr == 1))
    structure = np.ones((3, 3))  # 8-connectivity; use ndimage default (4-conn) if diagonal touches shouldn't merge sites
    labeled, n_sites = ndimage.label(is_hq, structure=structure)
    p.L.info(f"Found {n_sites} discrete high-quality sites "
              f"({int(is_hq.sum())} HQ pixels, {is_hq.sum()/max(n_sites,1):.1f} px/site avg)")

    pygeoprocessing.new_raster_from_base(input_path, output_path, gdal.GDT_Int32, [0])
    out_ds = gdal.Open(output_path, gdal.GA_Update)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(labeled.astype(np.int32))
    out_band.FlushCache()
    out_band, out_ds = None, None

    site_ids, pixel_counts = np.unique(labeled[labeled > 0], return_counts=True)
    pd.DataFrame({'site_id': site_ids, 'pixel_count': pixel_counts}).to_csv(site_sizes_path, index=False)
    p.L.info(f"Saved dissolved site raster to {output_path}")
    p.L.info(f"Saved site pixel-count table to {site_sizes_path}")


def recreation_site_analysis(p):
    """
    1) Combine environment index and accessibility index to rank recreation sites.
    2) Identify high quality recreation sites.
    3) Dissolve contiguous HQ pixels into discrete site blobs.
    """
    rank_rec_sites(p)
    find_high_quality_rec_sites(p)
    merge_hq_sites(p)


# ===========================================================================
# Overnight allocation (UNCHANGED from v1)
# ===========================================================================

def allocate_overnights(p):
    p.L.info("Allocating national overnight stays based on hotel points.")
    overnight_path = os.path.join(p.data_dir, 'gep_recreation/0_inputs/unwto-all-data-download_2022.xlsx')
    country_gpkg_path = os.path.join(p.data_dir, 'gep_recreation/0_inputs/ee_r250_correspondence.gpkg')
    ref_raster_path = os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/grip4_road_length/grip4_road_length_1km.tif')
    hotel_raster_path = os.path.join(p.cur_dir, 'hotels_1km.tif')
    output_path = os.path.join(p.cur_dir, 'overnights_1km.tif')

    if os.path.exists(output_path):
        p.L.info(f"Overnight raster already exists: {output_path}")
        return

    p.L.info("Rasterizing hotel data...")
    rasterize_hotel_points(p, hotel_raster_path)
    p.L.info("Importing UNTWO overnight data...")
    overnight_df = clean_untwo_data(p, overnight_path)
    p.L.info("Rasterizing country borders...")
    country_raster_path = rasterize_country_borders(p, country_gpkg_path, 'iso3_r250_ids_1km.tif', ref_raster_path, 'iso3_r250_id')
    substitution_report_path = os.path.join(p.cur_dir, 'overnights_year_substitution.csv')
    missing_overnights_report_path = os.path.join(p.cur_dir, 'missing_overnights_iso3.csv')
    create_overnight_allocation_raster(p, overnight_df, hotel_raster_path, country_raster_path, output_path,
                                        substitution_report_path=substitution_report_path,
                                        missing_overnights_report_path=missing_overnights_report_path)
    p.L.info(f"Overnight allocation complete: {output_path}")


def rasterize_hotel_points(p, hotel_raster_path):
    p.L.info("Rasterizing hotel points.")
    input_path = os.path.join(p.data_dir, 'gep_recreation/0_inputs/Export_GEO-data_4-University-of-Minnesota.gpkg')
    ref_path = os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/grip4_road_length/grip4_road_length_1km.tif')
    if os.path.exists(hotel_raster_path):
        p.L.info(f"Hotel raster already exists: {hotel_raster_path}")
        return

    ref_info = pygeoprocessing.geoprocessing.get_raster_info(ref_path)
    pygeoprocessing.geoprocessing.create_raster_from_bounding_box(
        target_bounding_box=ref_info['bounding_box'], target_raster_path=hotel_raster_path,
        target_pixel_size=ref_info['pixel_size'], target_pixel_type=gdal.GDT_Int16,
        target_srs_wkt=ref_info['projection_wkt'], target_nodata=-1, fill_value=0)
    pygeoprocessing.geoprocessing.rasterize(
        vector_path=input_path, target_raster_path=hotel_raster_path,
        burn_values=[1], option_list=['ALL_TOUCHED=TRUE', 'MERGE_ALG=ADD'])


def extract_clean_overnights(df, tourism_type):
    df = df.copy()
    df.columns = df.columns.map(str)
    iso3_r250_id_col, country_col = "C.", "Basic data and indicators"
    indicator_col, type_col = "Unnamed: 6", "Unnamed: 5"
    df[iso3_r250_id_col] = df[iso3_r250_id_col].ffill()
    df[country_col] = df[country_col].ffill()
    df[indicator_col] = df[indicator_col].ffill()
    df[type_col] = df[type_col].ffill()

    overnight_df = df[df[indicator_col].str.contains("Overnights", na=False)]
    year_cols = [col for col in df.columns if col.isdigit()]
    overnight_df = overnight_df[[iso3_r250_id_col, country_col, type_col] + year_cols]

    tidy = overnight_df.melt(id_vars=[iso3_r250_id_col, country_col, type_col],
                              var_name="year", value_name="overnights")
    tidy.rename(columns={iso3_r250_id_col: "iso3_r250_id", country_col: "unwto_name",
                          type_col: "overnight_type"}, inplace=True)
    tidy["tourism_type"] = tourism_type
    tidy["year"] = tidy["year"].astype(int)
    tidy["overnights"] = pd.to_numeric(tidy["overnights"], errors="coerce")
    return tidy


def clean_untwo_data(p, overnight_path):
    domestic_start = pd.read_excel(overnight_path, sheet_name='Domestic Tourism-Accommodation')
    inbound_start = pd.read_excel(overnight_path, sheet_name='Inbound Tourism-Accommodation')
    domestic_start_idx = domestic_start[domestic_start.iloc[:, 3] == 'AFGHANISTAN'].index[0]
    inbound_start_idx = inbound_start[inbound_start.iloc[:, 3] == 'AFGHANISTAN'].index[0]
    domestic_data = pd.read_excel(overnight_path, sheet_name='Domestic Tourism-Accommodation', skiprows=domestic_start_idx)
    inbound_data = pd.read_excel(overnight_path, sheet_name='Inbound Tourism-Accommodation', skiprows=inbound_start_idx)

    domestic_df = extract_clean_overnights(domestic_data, "domestic")
    inbound_df = extract_clean_overnights(inbound_data, "international")
    panel_df = pd.concat([domestic_df, inbound_df], ignore_index=True)
    panel_df["overnight_type"] = panel_df["overnight_type"].str.lower().str.strip()

    pivoted = panel_df.pivot_table(
        index=["iso3_r250_id", "unwto_name", "year"], columns=["overnight_type", "tourism_type"],
        values="overnights", aggfunc="first").reset_index()
    pivoted.columns = ['iso3_r250_id', 'unwto_name', 'year'] + [
        f"{otype}_overnights_{ttype}" for otype, ttype in pivoted.columns[3:]]

    for ttype in ['domestic', 'international']:
        total_col, hotel_col, combined_col = f"total_overnights_{ttype}", f"hotels and similar establishments_overnights_{ttype}", f"overnights_{ttype}"
        pivoted[combined_col] = np.where(pivoted[total_col].notna(), pivoted[total_col], pivoted[hotel_col])

    final_cols = ['iso3_r250_id', 'unwto_name', 'year']
    for ttype in ['domestic', 'international']:
        final_cols += [f"total_overnights_{ttype}", f"hotel_overnights_{ttype}", f"overnights_{ttype}"]
    pivoted.rename(columns={
        'hotels and similar establishments_overnights_domestic': 'hotel_overnights_domestic',
        'hotels and similar establishments_overnights_international': 'hotel_overnights_international'
    }, inplace=True)

    final_cols = [c for c in final_cols if c in pivoted.columns]
    final_df = pivoted[final_cols]
    final_df.to_csv(os.path.join(p.cur_dir, "unwto_panel.csv"), index=False)
    return final_df


def rasterize_country_borders(p, vector_path, raster_name, ref_raster_path, column):
    output_path = os.path.join(p.cur_dir, raster_name)
    if os.path.exists(output_path):
        p.L.info(f"Rasterized borders already exists: {output_path}")
        return output_path

    ref_info = pygeoprocessing.geoprocessing.get_raster_info(ref_raster_path)
    pygeoprocessing.geoprocessing.create_raster_from_bounding_box(
        target_bounding_box=ref_info['bounding_box'], target_raster_path=output_path,
        target_pixel_size=ref_info['pixel_size'], target_pixel_type=ref_info['datatype'],
        target_srs_wkt=ref_info['projection_wkt'], target_nodata=ref_info['nodata'][0], fill_value=0)
    pygeoprocessing.geoprocessing.rasterize(
        vector_path=vector_path, target_raster_path=output_path,
        burn_values=None, option_list=['ALL_TOUCHED=TRUE', f'ATTRIBUTE={column}'])
    return output_path


def create_overnight_allocation_raster(p, overnight_df, hotel_raster_path, country_raster_path, output_path,
                                        target_year=2019, substitution_report_path=None,
                                        missing_overnights_report_path=None):
    """
    Build the country -> total_overnights map, falling back to the nearest
    year with data when a country has no row for target_year (UNWTO
    reporting is patchy: many countries report exact years like 2019
    inconsistently, or stop reporting the "Overnights" indicator years
    before the panel ends -- see substitution_report_path for exactly which
    countries used a substitute year, how far off it was, or had NO usable
    year at all, so those results can be weighted/excluded downstream).
    """
    p.L.info("Creating country-to-overnight mapping (nearest-year fallback to target_year)...")
    df = overnight_df.copy()
    df['total_overnights'] = df[['overnights_domestic', 'overnights_international']].sum(axis=1, min_count=1)
    df = df[df['total_overnights'] > 0]

    country_overnights_map, substitution_rows = {}, []
    for iso3_r250_id, sub in df.groupby('iso3_r250_id'):
        sub = sub.assign(year_distance=(sub['year'] - target_year).abs())
        best = sub.sort_values(['year_distance', 'year'], ascending=[True, False]).iloc[0]
        country_overnights_map[iso3_r250_id] = best['total_overnights']
        substitution_rows.append({
            'iso3_r250_id': iso3_r250_id, 'unwto_name': best['unwto_name'],
            'year_used': int(best['year']), 'target_year': target_year,
            'year_distance': int(best['year_distance']), 'total_overnights': best['total_overnights'],
        })

    total_overnights_allocated = sum(country_overnights_map.values())
    exact_year_count = sum(1 for r in substitution_rows if r['year_distance'] == 0)
    p.L.info(f"Successfully mapped overnight data for {len(country_overnights_map)} countries "
             f"({exact_year_count} with exact {target_year} data, "
             f"{len(country_overnights_map) - exact_year_count} using a substitute year)")
    p.L.info(f"Total overnight stays allocated: {total_overnights_allocated:,.0f}")

    if substitution_report_path is not None:
        pd.DataFrame(substitution_rows).sort_values('year_distance', ascending=False).to_csv(
            substitution_report_path, index=False)
        p.L.info(f"Saved year-substitution QA report to {substitution_report_path} "
                 "(any country with no 'Overnights' data in ANY year is absent from this report "
                 "entirely and gets 0 overnight stays -- e.g. countries that only report "
                 "'Arrivals' to UNWTO, not overnight-stay counts)")

    if missing_overnights_report_path is not None:
        _, uncovered_overnights_iso3 = _log_overnights_coverage(p, country_raster_path, country_overnights_map)
        pd.DataFrame({'iso3_r250_id': sorted(uncovered_overnights_iso3)}).to_csv(
            missing_overnights_report_path, index=False)

    p.L.info("Allocating overnights to hotel pixels...")

    def allocate_overnights_op(hotel_array, country_array):
        result = np.zeros_like(hotel_array, dtype=np.float32)
        for ee_id, total_overnights in country_overnights_map.items():
            country_mask = (country_array == ee_id)
            hotels_in_country = np.sum(hotel_array[country_mask])
            if hotels_in_country > 0:
                result[country_mask] = (hotel_array[country_mask] / hotels_in_country) * total_overnights
        return result

    pygeoprocessing.raster_calculator([(hotel_raster_path, 1), (country_raster_path, 1)],
                                       allocate_overnights_op, output_path, gdal.GDT_Float32, -9999)
    p.L.info(f"Overnight allocation raster created: {output_path}")


def convert_pixel_size_to_km(raster_path, pixel_size_native):
    info = pygeoprocessing.get_raster_info(raster_path)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(info['projection_wkt'])
    if srs.IsGeographic():
        bbox = info['bounding_box']
        center_lat = (bbox[1] + bbox[3]) / 2.0
        km_per_degree_lat = 111.32
        km_per_degree_lon = 111.32 * np.cos(np.radians(center_lat))
        return pixel_size_native * (km_per_degree_lat + km_per_degree_lon) / 2.0
    else:
        linear_unit_name = srs.GetLinearUnitsName()
        if linear_unit_name and 'kilomet' in linear_unit_name.lower():
            return pixel_size_native
        return pixel_size_native / 1000.0


# ===========================================================================
# NEW v2: param-table-driven flow model
# ===========================================================================

def load_recreation_params(p, param_table_path=None):
    """
    Load and validate the external parameter table.

    Required columns: iso3_r250_id, iso3_r250_name, param_group,
                       participation_param, distance_param

    Returns:
      country_to_group_map : dict, iso3_r250_id -> param_group
      group_params          : dict, param_group -> {'a': participation_param,
                                                      'b': distance_param}
    Raises ValueError if any param_group has inconsistent parameter values
    across the countries assigned to it (that inconsistency would mean the
    grouping and the parameters disagree about what a "group" is).
    """
    if param_table_path is None:
        param_table_path = os.path.join(p.data_dir, 'gep_recreation/0_inputs/recreation_demand_params_global.csv')

    df = pd.read_csv(param_table_path)
    required = ['iso3_r250_id', 'iso3_r250_name', 'param_group', 'participation_param', 'distance_param']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Parameter table missing required columns: {missing_cols}")

    # validate: every row sharing a param_group must agree on (a, b)
    check = df.groupby('param_group')[['participation_param', 'distance_param']].nunique()
    bad_groups = check[(check['participation_param'] > 1) | (check['distance_param'] > 1)]
    if not bad_groups.empty:
        raise ValueError(
            "param_group(s) with inconsistent participation_param/distance_param "
            f"across countries: {list(bad_groups.index)}. Every country sharing a "
            "param_group must have identical parameter values.")

    country_to_group_map = dict(zip(df['iso3_r250_id'].astype(int), df['param_group']))
    group_params = {}
    for gid, sub in df.groupby('param_group'):
        row = sub.iloc[0]
        group_params[int(gid)] = {'a': float(row['participation_param']), 'b': float(row['distance_param'])}

    p.L.info(f"Loaded recreation demand parameters: {len(group_params)} param_group(s), "
             f"{len(country_to_group_map)} countries")
    return country_to_group_map, group_params


def build_group_id_raster(country_id_raster_path, country_to_group_map, output_path):
    """Remap a country-id raster into a param_group id raster."""
    if os.path.exists(output_path):
        return output_path
    country_info = pygeoprocessing.get_raster_info(country_id_raster_path)
    country_nodata = country_info['nodata'][0]
    group_nodata = -1

    max_country_id = max(country_to_group_map.keys()) if country_to_group_map else 0
    lookup = np.full(max_country_id + 1, group_nodata, dtype=np.int32)
    for cid, gid in country_to_group_map.items():
        if 0 <= cid <= max_country_id:
            lookup[cid] = gid

    def remap_op(country_block):
        out = np.full_like(country_block, group_nodata, dtype=np.int32)
        valid = (country_block != country_nodata) & (country_block >= 0) & (country_block <= max_country_id) \
            if country_nodata is not None else ((country_block >= 0) & (country_block <= max_country_id))
        out[valid] = lookup[country_block[valid].astype(int)]
        return out

    pygeoprocessing.raster_calculator([(country_id_raster_path, 1)], remap_op, output_path,
                                       gdal.GDT_Int32, group_nodata)
    return output_path


def _write_array_as_raster(array, pixel_size_km, output_path, nodata=-9999):
    """Write a plain numpy array to a small local GeoTIFF -- used both for
    kernels (f(d), d*f(d), sampled at pixel offsets) and for per-group Q
    rasters fed into pygeoprocessing.convolve_2d, which requires file-based
    inputs. The geotransform is a local placeholder: convolve_2d aligns by
    pixel count, so only the pixel size (matching pixel_size_km) matters,
    not real-world placement."""
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = array.shape
    # TILED=YES forces square blocks (default 256x256); without it GDAL
    # defaults to row-strip layout, which pygeoprocessing.convolve_2d
    # rejects as "very slow" and raises rather than silently accepting.
    ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32,
                        options=['TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'COMPRESS=LZW'])
    ds.SetGeoTransform((0, pixel_size_km, 0, 0, 0, -pixel_size_km))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(array.astype(np.float32))
    band.SetNoDataValue(nodata)
    band.FlushCache()
    ds = None


def compute_group_bands(pixel_size_km, max_distance_km=50.0, max_bands=8):
    """
    Fixed-cutoff banding, grid-aware:
      cutoff_km  = max_distance_km            -- a manually-set farthest
                   reasonable day-trip distance, the SAME for every param
                   group regardless of that group's own decay rate b.
                   (Previously this was derived as cutoff_k_multiplier / b,
                   which scaled with each group's own typical trip range;
                   that's no longer used because a distance-decay-derived
                   cutoff has no necessary relationship to how far someone
                   would plausibly travel and return same-day -- b shapes
                   the SPLIT of demand across nearby sites, but shouldn't
                   also be trusted to say where demand physically stops.)
      n_bands    = floor(cutoff_km / pixel_size_km), clamped to [1, max_bands]
                   -- guarantees every band is at least one pixel wide. A
                   band narrower than the grid can resolve isn't measuring
                   anything real; this collapses gracefully to n_bands=1
                   for a cutoff that's already sub-grid-scale.
      band_width = cutoff_km / n_bands        -- NOT snapped to the discrete
                   set of achievable pixel-offset radii; band membership is
                   therefore approximate at each band's edge, traded for
                   much simpler, uniform-width bands per group.

    b is still used elsewhere (cdf_exp) to shape how much of the fixed
    cutoff's demand falls in each band; it just no longer sets the cutoff
    itself.

    Returns (cutoff_km, edges) where edges is a list of n_bands+1 km values
    from 0 to cutoff_km.
    """
    cutoff_km = max_distance_km
    n_bands = int(np.clip(np.floor(cutoff_km / pixel_size_km), 1, max_bands))
    band_width = cutoff_km / n_bands
    edges = [i * band_width for i in range(n_bands + 1)]
    return cutoff_km, edges


def cdf_exp(d_km, b):
    """CDF of the normalized exponential density f(d) = b*exp(-b*d).
    Closed form -- no numerical integration needed for band shares."""
    return 1.0 - np.exp(-b * d_km)


def build_ring_kernel(r_lo, r_hi, pixel_size_km, kernel_dir, tag):
    """
    Binary indicator kernel: 1 for pixel offsets whose distance falls in
    (r_lo, r_hi] (or [0, r_hi] for the first band, r_lo=0), else 0. Used
    for BOTH detecting site presence within the band (denom) and spreading
    a band's fixed, pre-allocated visit/value pool evenly across whatever
    sites are present -- since within-band allocation is a flat/uniform
    split (band price uses the band midpoint, not per-site distance), one
    kernel per band covers every convolution that band needs.
    """
    radius_px = int(np.ceil(r_hi / pixel_size_km))
    yy, xx = np.mgrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    d_km = np.sqrt(xx**2 + yy**2) * pixel_size_km
    within = (d_km <= r_hi) if r_lo == 0 else ((d_km > r_lo) & (d_km <= r_hi))
    kernel = within.astype(np.float32)
    kernel_path = os.path.join(kernel_dir, f'ring_kernel_{tag}.tif')
    _write_array_as_raster(kernel, pixel_size_km, kernel_path)
    return kernel_path


def _scan_group_bboxes(group_id_path, group_nodata=-1):
    """
    One streaming pass over the group_id raster (via iterblocks -- never
    holds the full raster in memory) to find each group's pixel bounding
    box. This is the only thing that needs a global look at group_id;
    everything else in calculate_recreation_flows reads windowed slices
    directly off disk for just one group's clipped extent at a time.

    Returns dict: group_id -> [row_min, row_max, col_min, col_max]
    (row_max/col_max are inclusive of the last seen pixel).
    """
    bboxes = {}
    ds = gdal.Open(group_id_path)
    band = ds.GetRasterBand(1)
    for block_info in pygeoprocessing.iterblocks((group_id_path, 1), offset_only=True):
        xoff, yoff = block_info['xoff'], block_info['yoff']
        wx, wy = block_info['win_xsize'], block_info['win_ysize']
        block = band.ReadAsArray(xoff, yoff, wx, wy)
        present = np.unique(block[block != group_nodata])
        for gid in present:
            gid = int(gid)
            rows, cols = np.where(block == gid)
            r0, r1 = rows.min() + yoff, rows.max() + yoff
            c0, c1 = cols.min() + xoff, cols.max() + xoff
            if gid not in bboxes:
                bboxes[gid] = [r0, r1, c0, c1]
            else:
                bboxes[gid][0] = min(bboxes[gid][0], r0)
                bboxes[gid][1] = max(bboxes[gid][1], r1)
                bboxes[gid][2] = min(bboxes[gid][2], c0)
                bboxes[gid][3] = max(bboxes[gid][3], c1)
    ds = None
    return bboxes


def _read_window(band, xoff, yoff, wx, wy):
    return band.ReadAsArray(xoff, yoff, wx, wy)


def _accumulate_window(band, xoff, yoff, window_contribution):
    """Read-modify-write add: needed because groups' buffered windows can
    overlap (population near a border contributes to sites just across
    it), so a later group's contribution to a shared window must be added
    to, not overwrite, an earlier group's. Kept to windows, not the full
    raster, so memory use stays bounded by cutoff_km/pixel_size regardless
    of global raster size."""
    existing = band.ReadAsArray(xoff, yoff, window_contribution.shape[1], window_contribution.shape[0])
    band.WriteArray(existing + window_contribution, xoff, yoff)


def _assert_rasters_aligned(reference_path, other_paths_by_name, tolerance=1e-6):
    """
    Raise a clear error if any raster doesn't share the reference raster's
    pixel grid (raster_size, pixel_size, and origin/bounding_box). This
    pipeline reuses one raster's (xoff, yoff, wx, wy) window to read
    windows out of several other rasters on the unchecked assumption they
    share a grid -- if they don't, reads silently return data from the
    wrong geographic location instead of erroring, which is much worse
    than a crash (see _write_site_table's "all zeros" symptom: a mask from
    one grid applied to values from a misaligned grid mostly lands on
    background pixels rather than raising anything).
    """
    ref_info = pygeoprocessing.get_raster_info(reference_path)
    ref_size = ref_info['raster_size']
    ref_pixel = ref_info['pixel_size']
    ref_bbox = ref_info['bounding_box']

    problems = []
    for name, path in other_paths_by_name.items():
        info = pygeoprocessing.get_raster_info(path)
        if info['raster_size'] != ref_size:
            problems.append(f"{name} ({path}): raster_size {info['raster_size']} != reference {ref_size}")
            continue
        if any(abs(a - b) > tolerance for a, b in zip(info['pixel_size'], ref_pixel)):
            problems.append(f"{name} ({path}): pixel_size {info['pixel_size']} != reference {ref_pixel}")
        if any(abs(a - b) > tolerance for a, b in zip(info['bounding_box'], ref_bbox)):
            problems.append(f"{name} ({path}): bounding_box {info['bounding_box']} != reference {ref_bbox}")

    if problems:
        raise ValueError(
            "Raster grid mismatch detected -- calculate_recreation_flows reuses "
            "one raster's pixel offsets to window-read the others, which silently "
            "returns geographically wrong data (not an error) if grids differ. "
            "Align these to the population raster's grid first, e.g. via "
            "pygeoprocessing.align_and_resize_raster_stack, before rerunning:\n"
            + "\n".join(problems))


def _log_coverage_summary(p, country_id_path, group_id_path, group_params):
    """
    Cross-references country_id against group_id (streaming, no full-raster
    load) to report which countries actually resolve to a valid param_group
    vs. which are present in the data but silently contribute zero visits
    because their group_id has no entry in group_params. This is the check
    that would have caught the "identical output after swapping in a
    global param table" bug immediately, rather than requiring several
    rounds of downstream symptom-chasing.
    """
    country_nodata = pygeoprocessing.get_raster_info(country_id_path)['nodata'][0]
    valid_group_ids = np.array(sorted(group_params.keys()))

    country_ds = gdal.Open(country_id_path); country_band = country_ds.GetRasterBand(1)
    group_ds = gdal.Open(group_id_path); group_band = group_ds.GetRasterBand(1)

    covered, uncovered = set(), set()
    for block_info in pygeoprocessing.iterblocks((country_id_path, 1), offset_only=True):
        xoff, yoff = block_info['xoff'], block_info['yoff']
        wx, wy = block_info['win_xsize'], block_info['win_ysize']
        country_block = country_band.ReadAsArray(xoff, yoff, wx, wy)
        group_block = group_band.ReadAsArray(xoff, yoff, wx, wy)
        valid = (country_block != country_nodata) & (country_block > 0) if country_nodata is not None \
            else (country_block > 0)
        if not np.any(valid):
            continue
        has_param = np.isin(group_block[valid], valid_group_ids)
        covered.update(np.unique(country_block[valid][has_param]).tolist())
        uncovered.update(np.unique(country_block[valid][~has_param]).tolist())
    uncovered -= covered  # a country with even one covered pixel isn't "missing"
    country_ds, group_ds = None, None

    p.L.info(f"Parameter coverage: {len(covered)} countries with valid param_group, "
             f"{len(uncovered)} countries present in country_id raster but with NO valid parameters "
             f"(contributing exactly zero visits/value/unmet)")
    if uncovered:
        p.L.warning(f"Uncovered iso3_r250_id values: {sorted(uncovered)}")
    return covered, uncovered


def _log_cost_coverage(p, country_id_path, cost_map):
    """
    Cross-references country_id against cost_map (streaming, no full-raster
    load) to report which countries have real fuel-cost data vs. which are
    present in the data but have no entry in cost_map. Countries with no
    cost data get cost_per_km=0 in calculate_recreation_flows, which would
    otherwise silently read as "travel is free here" (value=0) rather than
    "cost is unknown" -- the two are not the same thing, so this list is
    used downstream (aggregate_results_iso3) to overwrite those countries'
    aggregated value with NaN instead of a spurious real zero.
    """
    country_nodata = pygeoprocessing.get_raster_info(country_id_path)['nodata'][0]
    valid_cost_ids = np.array(sorted(cost_map.keys()))

    country_ds = gdal.Open(country_id_path); country_band = country_ds.GetRasterBand(1)

    covered, uncovered = set(), set()
    for block_info in pygeoprocessing.iterblocks((country_id_path, 1), offset_only=True):
        xoff, yoff = block_info['xoff'], block_info['yoff']
        wx, wy = block_info['win_xsize'], block_info['win_ysize']
        country_block = country_band.ReadAsArray(xoff, yoff, wx, wy)
        valid = (country_block != country_nodata) & (country_block > 0) if country_nodata is not None \
            else (country_block > 0)
        if not np.any(valid):
            continue
        has_cost = np.isin(country_block[valid], valid_cost_ids)
        covered.update(np.unique(country_block[valid][has_cost]).tolist())
        uncovered.update(np.unique(country_block[valid][~has_cost]).tolist())
    uncovered -= covered
    country_ds = None

    p.L.info(f"Fuel-cost coverage: {len(covered)} countries with real cost data, "
             f"{len(uncovered)} countries present in country_id raster but with NO cost data "
             f"(value computed as if travel is free there -- flagged for NaN downstream)")
    if uncovered:
        p.L.warning(f"iso3_r250_id values with missing fuel-cost data: {sorted(uncovered)}")
    return covered, uncovered


def _log_overnights_coverage(p, country_id_path, country_overnights_map):
    """
    Cross-references country_id against country_overnights_map (streaming,
    no full-raster load) to report which countries have NO usable UNWTO
    "Overnights" data in ANY year (not just target_year -- see
    create_overnight_allocation_raster's nearest-year fallback). Those
    countries get an overnights raster of all zeros, which reads as "zero
    tourist demand" rather than "tourism data unavailable" -- e.g. Indonesia
    reports Arrivals to UNWTO but never Overnights, so it would otherwise
    silently look identical to a country with genuinely no tourism.
    """
    country_nodata = pygeoprocessing.get_raster_info(country_id_path)['nodata'][0]
    valid_overnight_ids = np.array(sorted(country_overnights_map.keys()))

    country_ds = gdal.Open(country_id_path); country_band = country_ds.GetRasterBand(1)

    covered, uncovered = set(), set()
    for block_info in pygeoprocessing.iterblocks((country_id_path, 1), offset_only=True):
        xoff, yoff = block_info['xoff'], block_info['yoff']
        wx, wy = block_info['win_xsize'], block_info['win_ysize']
        country_block = country_band.ReadAsArray(xoff, yoff, wx, wy)
        valid = (country_block != country_nodata) & (country_block > 0) if country_nodata is not None \
            else (country_block > 0)
        if not np.any(valid):
            continue
        has_overnights = np.isin(country_block[valid], valid_overnight_ids)
        covered.update(np.unique(country_block[valid][has_overnights]).tolist())
        uncovered.update(np.unique(country_block[valid][~has_overnights]).tolist())
    uncovered -= covered
    country_ds = None

    p.L.info(f"Overnights coverage: {len(covered)} countries with usable UNWTO Overnights data "
             f"(any year), {len(uncovered)} countries present in country_id raster but with NO "
             f"Overnights data in any year (tourist demand there reads as a real 0 -- flagged for "
             f"NaN downstream)")
    if uncovered:
        p.L.warning(f"iso3_r250_id values with NO Overnights data in any year: {sorted(uncovered)}")
    return covered, uncovered


def calculate_recreation_flows(p, inputs, outputs, group_params, max_distance_km=50.0, max_bands=8):
    """
    Core v2 flow model, banded version. For every param_group g with (a, b):
      cutoff_km_g = max_distance_km     (fixed farthest reasonable day-trip
                     distance, the same for every group -- see
                     compute_group_bands for why this is no longer derived
                     from b)
      bands: cutoff_km_g divided into 1-8 pixel-grid-aware rings (see
             compute_group_bands / build_ring_kernel)

    Per population pixel i, per band k with (r_lo, r_hi):
      band_share_k = CDF(r_hi) - CDF(r_lo)      where CDF(d)=1-exp(-b*d)
      Q_band_i     = pop_i * a * band_share_k    (fixed budget for this ring,
                     independent of how many sites happen to sit inside it)
      if >=1 site reachable within the ring: split Q_band_i evenly across
        them (uniform, not distance-weighted -- band price uses the ring
        midpoint, not per-site distance)
      if 0 sites reachable within the ring: Q_band_i is entirely unmet
      value uses the ring's midpoint distance, round-trip, at the ORIGIN
        pixel's country cost_per_km (p*q, not consumer surplus)
      beyond cutoff_km_g: (1 - CDF(cutoff_km_g)) * pop_i * a is unmet,
        unconditionally (same role the old fixed cutoff played, now scaled
        per group)

    This replaces the earlier single-smooth-kernel version specifically to
    avoid a form of the original double-counting bug re-appearing at the
    ring level: several sites clustered within the same distance range
    would otherwise each draw a full, independent f(d) weight rather than
    sharing one fixed per-ring allocation.

    MEMORY: this version never loads a full global raster into memory.
    Only group_id gets one streaming pass (_scan_group_bboxes, to find each
    group's extent) -- pop/dist/country/hq are all read as small WINDOWED
    slices, one group's clipped bounding box at a time, directly off disk.

    inputs: dict with 'hq_sites', 'hq_sites_dissolved', 'population',
            'group_id', 'country_id', 'country_costs' (csv path)
    outputs: dict with 'visits', 'value', 'unmet' (raster paths),
             'site_table', 'group_table' (csv paths)
    """
    work_dir = os.path.join(p.cur_dir, 'flows')
    kernel_dir = os.path.join(work_dir, 'kernels')
    os.makedirs(kernel_dir, exist_ok=True)

    p.L.info("Verifying all input rasters share the population raster's pixel grid")
    _assert_rasters_aligned(inputs['population'], {
        'hq_sites': inputs['hq_sites'],
        'group_id': inputs['group_id'],
        'country_id': inputs['country_id'],
    })

    raster_info = pygeoprocessing.get_raster_info(inputs['population'])
    px_x, px_y = abs(raster_info['pixel_size'][0]), abs(raster_info['pixel_size'][1])
    if abs(px_x - px_y) > 1e-9 * max(px_x, px_y):
        raise ValueError(
            f"Population raster has non-square pixels ({px_x} vs {px_y}); the ring-kernel "
            "distance geometry in this module assumes a single pixel_size_km for both axes.")
    pixel_size_native = px_x
    pixel_size_km = convert_pixel_size_to_km(inputs['population'], pixel_size_native)
    pop_nodata = raster_info['nodata'][0]
    n_cols, n_rows = raster_info['raster_size']

    cost_df = pd.read_csv(inputs['country_costs'])
    cost_map = dict(zip(cost_df['iso3_r250_id'], cost_df['gasoline_cost_usd_per_km_2019_gppdata']))
    max_cid = max(cost_map.keys()) if cost_map else 0
    cost_lookup = np.zeros(max_cid + 1, dtype=np.float64)
    for cid, cost in cost_map.items():
        if 0 <= cid <= max_cid:
            cost_lookup[cid] = cost

    dist_path = os.path.join(work_dir, 'distance_to_hq_km.tif')
    if os.path.exists(dist_path):
        p.L.info(f"Distance raster already exists: {dist_path}")
    else:
        p.L.info("Computing distance to nearest high-quality site")
        pygeoprocessing.distance_transform_edt(
            (inputs['hq_sites'], 1), dist_path,
            sampling_distance=(pixel_size_km, pixel_size_km), working_dir=work_dir)

    hq_nodata = pygeoprocessing.get_raster_info(inputs['hq_sites'])['nodata'][0]

    # --- output rasters: created once, filled with 0 (not nodata) so that
    # windowed read-modify-write ADD works correctly across overlapping
    # group buffers. 0 here means "no visits/value/unmet ever landed on
    # this pixel", which for a site-flow raster is the natural background
    # value -- simpler than juggling nodata-vs-zero during accumulation.
    for key in ['visits', 'value', 'unmet']:
        pygeoprocessing.new_raster_from_base(
            inputs['population'], outputs[key], gdal.GDT_Float32, [-9999])
        ds = gdal.Open(outputs[key], gdal.GA_Update)
        ds.GetRasterBand(1).Fill(0)
        ds = None

    p.L.info("Scanning parameter-group extents")
    bboxes = _scan_group_bboxes(inputs['group_id'])
    p.L.info(f"Processing {len(bboxes)} parameter group(s)")

    _log_coverage_summary(p, inputs['country_id'], inputs['group_id'], group_params)
    _, uncovered_cost_iso3 = _log_cost_coverage(p, inputs['country_id'], cost_map)
    pd.DataFrame({'iso3_r250_id': sorted(uncovered_cost_iso3)}).to_csv(
        outputs['missing_cost_iso3'], index=False)

    group_qa_rows = []

    # persistent, reused across groups -- windowed reads only
    pop_ds = gdal.Open(inputs['population']); pop_band = pop_ds.GetRasterBand(1)
    dist_ds = gdal.Open(dist_path); dist_band = dist_ds.GetRasterBand(1)
    country_ds = gdal.Open(inputs['country_id']); country_band = country_ds.GetRasterBand(1)
    hq_ds = gdal.Open(inputs['hq_sites']); hq_band = hq_ds.GetRasterBand(1)
    group_ds = gdal.Open(inputs['group_id']); group_band = group_ds.GetRasterBand(1)

    visits_out_ds = gdal.Open(outputs['visits'], gdal.GA_Update); visits_out_band = visits_out_ds.GetRasterBand(1)
    value_out_ds = gdal.Open(outputs['value'], gdal.GA_Update); value_out_band = value_out_ds.GetRasterBand(1)
    unmet_out_ds = gdal.Open(outputs['unmet'], gdal.GA_Update); unmet_out_band = unmet_out_ds.GetRasterBand(1)

    for gid, (r0, r1, c0, c1) in bboxes.items():
        params = group_params.get(gid)
        if params is None:
            p.L.warning(f"No parameters for param_group {gid}, skipping")
            continue
        a, b = params['a'], params['b']

        cutoff_km_g, edges = compute_group_bands(pixel_size_km, max_distance_km, max_bands)
        buffer_px_g = int(np.ceil(cutoff_km_g / pixel_size_km))

        # expand bbox by THIS GROUP's own cutoff buffer, clamp to raster bounds
        r0b, r1b = max(r0 - buffer_px_g, 0), min(r1 + buffer_px_g + 1, n_rows)
        c0b, c1b = max(c0 - buffer_px_g, 0), min(c1 + buffer_px_g + 1, n_cols)
        xoff, yoff = c0b, r0b
        wx, wy = c1b - c0b, r1b - r0b

        group_sub = _read_window(group_band, xoff, yoff, wx, wy)
        pop_sub = _read_window(pop_band, xoff, yoff, wx, wy).astype(np.float64)
        country_sub = _read_window(country_band, xoff, yoff, wx, wy)
        hq_sub = _read_window(hq_band, xoff, yoff, wx, wy)

        group_mask = (group_sub == gid)
        valid_sub = (pop_sub != pop_nodata) & ~np.isnan(pop_sub) & (pop_sub > 0) & group_mask

        budget = np.where(valid_sub, pop_sub * a, 0.0)
        cost_per_km_sub = np.zeros_like(pop_sub, dtype=np.float64)
        in_range = valid_sub & (country_sub >= 0) & (country_sub <= max_cid)
        cost_per_km_sub[in_range] = cost_lookup[country_sub[in_range].astype(int)]

        site_ind_sub = (hq_sub == 1).astype(np.float32) if hq_nodata is None \
            else ((hq_sub != hq_nodata) & (hq_sub == 1)).astype(np.float32)

        tmp_dir = os.path.join(kernel_dir, f'tmp_{gid}')
        os.makedirs(tmp_dir, exist_ok=True)
        site_ind_path = os.path.join(tmp_dir, 'site_ind.tif')
        _write_array_as_raster(site_ind_sub, pixel_size_km, site_ind_path)

        realized_total = np.zeros_like(pop_sub, dtype=np.float64)
        unmet_total = np.zeros_like(pop_sub, dtype=np.float64)
        visits_accum_g = np.zeros_like(pop_sub, dtype=np.float64)
        value_accum_g = np.zeros_like(pop_sub, dtype=np.float64)

        n_bands = len(edges) - 1
        for k in range(n_bands):
            r_lo, r_hi = edges[k], edges[k + 1]
            r_mid = 0.5 * (r_lo + r_hi)
            band_share = cdf_exp(r_hi, b) - cdf_exp(r_lo, b)

            Q_band = np.where(valid_sub, budget * band_share, 0.0)

            ring_kernel_path = build_ring_kernel(r_lo, r_hi, pixel_size_km, kernel_dir, tag=f'{gid}_{k}')
            denom_path = os.path.join(tmp_dir, f'denom_{k}.tif')
            q_visits_path = os.path.join(tmp_dir, f'q_visits_{k}.tif')
            q_value_path = os.path.join(tmp_dir, f'q_value_{k}.tif')
            visits_conv_path = os.path.join(tmp_dir, f'visits_at_sites_{k}.tif')
            value_conv_path = os.path.join(tmp_dir, f'value_at_sites_{k}.tif')

            # denom_k_i = count of sites reachable within this ring
            pygeoprocessing.convolve_2d(
                (site_ind_path, 1), (ring_kernel_path, 1), denom_path,
                ignore_nodata_and_edges=True, mask_nodata=False,
                target_datatype=gdal.GDT_Float32, target_nodata=0)
            denom_ds = gdal.Open(denom_path)
            denom_arr = denom_ds.GetRasterBand(1).ReadAsArray()
            denom_ds = None
            denom_safe = np.where(denom_arr > 1e-12, denom_arr, np.nan)

            # split this ring's fixed budget evenly across sites present;
            # 0 sites in ring -> this ring's whole share is unmet
            Q_visits_norm = np.where(np.isfinite(denom_safe), Q_band / denom_safe, 0.0).astype(np.float32)
            Q_value_norm = np.where(np.isfinite(denom_safe),
                                     (Q_band / denom_safe) * cost_per_km_sub * 2.0 * r_mid, 0.0).astype(np.float32)

            no_site_in_ring = ~np.isfinite(denom_safe) & valid_sub
            unmet_total[no_site_in_ring] += Q_band[no_site_in_ring]
            realized_total += np.where(no_site_in_ring, 0.0, Q_band)

            _write_array_as_raster(Q_visits_norm, pixel_size_km, q_visits_path)
            _write_array_as_raster(Q_value_norm, pixel_size_km, q_value_path)

            pygeoprocessing.convolve_2d(
                (q_visits_path, 1), (ring_kernel_path, 1), visits_conv_path,
                ignore_nodata_and_edges=True, mask_nodata=False,
                target_datatype=gdal.GDT_Float32, target_nodata=0)
            pygeoprocessing.convolve_2d(
                (q_value_path, 1), (ring_kernel_path, 1), value_conv_path,
                ignore_nodata_and_edges=True, mask_nodata=False,
                target_datatype=gdal.GDT_Float32, target_nodata=0)

            visits_ds = gdal.Open(visits_conv_path)
            visits_accum_g += visits_ds.GetRasterBand(1).ReadAsArray() * site_ind_sub
            visits_ds = None
            value_ds = gdal.Open(value_conv_path)
            value_accum_g += value_ds.GetRasterBand(1).ReadAsArray() * site_ind_sub
            value_ds = None

        # beyond this group's cutoff: unconditionally unmet, same role the
        # old fixed global cutoff played, now scaled to this group's own b
        beyond_cutoff_share = 1.0 - cdf_exp(cutoff_km_g, b)
        beyond_cutoff_unmet = np.where(valid_sub, budget * beyond_cutoff_share, 0.0)
        unmet_total += beyond_cutoff_unmet

        group_qa_rows.append({
            'param_group': gid, 'participation_param_a': a, 'distance_param_b': b,
            'cutoff_km': cutoff_km_g, 'n_bands': n_bands,
            'budget_total': float(budget.sum()),
            'realized_total': float(realized_total.sum()),
            'unmet_total': float(unmet_total.sum()),
        })

        _accumulate_window(visits_out_band, xoff, yoff, visits_accum_g.astype(np.float32))
        _accumulate_window(value_out_band, xoff, yoff, value_accum_g.astype(np.float32))
        _accumulate_window(unmet_out_band, xoff, yoff, unmet_total.astype(np.float32))

        # free this group's temp files now rather than at the very end --
        # avoids accumulating hundreds of groups' worth of clipped rasters
        # on disk simultaneously (the earlier "no space left on device" failure)
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    for band in [visits_out_band, value_out_band, unmet_out_band]:
        band.FlushCache()
    # explicit per-name closure -- `for ds in [...]: ds = None` does NOT
    # close anything (it only rebinds the loop variable), which left
    # visits_out_ds/value_out_ds/unmet_out_ds technically still open in
    # GA_Update mode when _write_site_table below opened a second,
    # read-only handle on the same paths. GDAL's GTiff driver can defer
    # finalizing tile/block directory metadata until the dataset object is
    # actually destroyed, not merely flushed -- so a concurrent read-only
    # open of the same file while the writer is still alive can silently
    # see an incomplete (all-zero) view, exactly matching the symptom
    # (pixel_count/site locations correct, values all 0.0) traced via
    # diagnose_site_table.py.
    pop_band = None; pop_ds = None
    dist_band = None; dist_ds = None
    country_band = None; country_ds = None
    hq_band = None; hq_ds = None
    group_band = None; group_ds = None
    visits_out_band = None; visits_out_ds = None
    value_out_band = None; value_out_ds = None
    unmet_out_band = None; unmet_out_ds = None

    pd.DataFrame(group_qa_rows).to_csv(outputs['group_table'], index=False)
    p.L.info(f"Saved param_group QA table (budget/realized/unmet) to {outputs['group_table']}")

    _write_site_table(inputs['hq_sites'], inputs['country_id'], outputs['visits'], outputs['value'],
                       outputs['site_table'], site_id_path=inputs.get('hq_sites_dissolved'))
    p.L.info(f"Saved site-level visits/value table to {outputs['site_table']}")


def _write_site_table(hq_sites_path, country_id_path, visits_path, value_path, output_path, site_id_path=None):
    """
    One row per dissolved site (if hq_sites_dissolved given) or per HQ
    pixel (fallback). Streams over blocks rather than loading full global
    rasters -- site pixels are a tiny fraction of the raster, so the
    per-block filtered results stay small even though the scan itself
    covers the whole extent.
    """
    hq_nodata = pygeoprocessing.get_raster_info(hq_sites_path)['nodata'][0]
    hq_ds = gdal.Open(hq_sites_path); hq_band = hq_ds.GetRasterBand(1)
    country_ds = gdal.Open(country_id_path); country_band = country_ds.GetRasterBand(1)
    visits_ds = gdal.Open(visits_path); visits_band = visits_ds.GetRasterBand(1)
    value_ds = gdal.Open(value_path); value_band = value_ds.GetRasterBand(1)
    site_id_ds, site_id_band = (None, None)
    if site_id_path and os.path.exists(site_id_path):
        site_id_ds = gdal.Open(site_id_path)
        site_id_band = site_id_ds.GetRasterBand(1)

    chunks = []
    for block_info in pygeoprocessing.iterblocks((hq_sites_path, 1), offset_only=True):
        xoff, yoff = block_info['xoff'], block_info['yoff']
        wx, wy = block_info['win_xsize'], block_info['win_ysize']

        hq_block = hq_band.ReadAsArray(xoff, yoff, wx, wy)
        is_hq = (hq_block == 1) if hq_nodata is None else ((hq_block != hq_nodata) & (hq_block == 1))
        if not np.any(is_hq):
            continue

        country_block = country_band.ReadAsArray(xoff, yoff, wx, wy)
        visits_block = visits_band.ReadAsArray(xoff, yoff, wx, wy)
        value_block = value_band.ReadAsArray(xoff, yoff, wx, wy)

        if site_id_band is not None:
            site_id_block = site_id_band.ReadAsArray(xoff, yoff, wx, wy)
            mask = is_hq & (site_id_block > 0)
            rows, cols = np.where(mask)
            chunks.append(pd.DataFrame({
                'site_id': site_id_block[rows, cols], 'iso3_r250_id': country_block[rows, cols],
                'visits': visits_block[rows, cols], 'value': value_block[rows, cols]}))
        else:
            rows, cols = np.where(is_hq)
            chunks.append(pd.DataFrame({
                'row': rows + yoff, 'col': cols + xoff, 'iso3_r250_id': country_block[rows, cols],
                'visits': visits_block[rows, cols], 'value': value_block[rows, cols]}))

    for ds in [hq_ds, country_ds, visits_ds, value_ds, site_id_ds]:
        ds = None
    has_site_id = site_id_band is not None
    hq_band, country_band, visits_band, value_band, site_id_band = None, None, None, None, None
    hq_ds, country_ds, visits_ds, value_ds, site_id_ds = None, None, None, None, None

    per_pixel = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=['site_id', 'iso3_r250_id', 'visits', 'value'])

    if has_site_id:
        agg = per_pixel.groupby('site_id').agg(
            visits=('visits', 'sum'), value=('value', 'sum'), pixel_count=('visits', 'size'),
            iso3_r250_id=('iso3_r250_id', lambda s: s.value_counts().idxmax())).reset_index()
        agg.to_csv(output_path, index=False)
    else:
        per_pixel.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Task-tree entry points (names kept from v1 for wiring compatibility)
# ---------------------------------------------------------------------------

def fit_recreation_demand_params(p):
    """Load + validate the external parameter table, build the aligned
    param_group id raster. Shared by both daily and tourist flows."""
    output_group_id_raster_path = os.path.join(p.cur_dir, 'group_id_1km.tif')
    if os.path.exists(output_group_id_raster_path):
        p.L.info("Parameter group raster already exists, skipping")
        return

    country_id_raster_path = os.path.join(p.allocate_overnights_dir, 'iso3_r250_ids_1km.tif')
    if not os.path.exists(country_id_raster_path):
        raise FileNotFoundError(f"Missing input file for fit_recreation_demand_params: {country_id_raster_path}")

    country_to_group_map, group_params = load_recreation_params(p)

    import pickle
    with open(os.path.join(p.cur_dir, 'group_params.pkl'), 'wb') as f:
        pickle.dump(group_params, f)

    build_group_id_raster(country_id_raster_path, country_to_group_map, output_group_id_raster_path)
    p.L.info(f"Saved param_group id raster to {output_group_id_raster_path}")


def _load_group_params(p):
    import pickle
    with open(os.path.join(p.fit_recreation_demand_params_dir, 'group_params.pkl'), 'rb') as f:
        return pickle.load(f)


def calculate_daily_recreation_into_sites_kernel(p):
    """Resident population -> site flows (visits, value, unmet demand)."""
    outputs = {
        'visits': os.path.join(p.cur_dir, 'site_visits_1km.tif'),
        'value': os.path.join(p.cur_dir, 'site_value_1km.tif'),
        'unmet': os.path.join(p.cur_dir, 'unmet_demand_1km.tif'),
        'site_table': os.path.join(p.cur_dir, 'site_visits_value.csv'),
        'group_table': os.path.join(p.cur_dir, 'param_group_qa.csv'),
        'missing_cost_iso3': os.path.join(p.cur_dir, 'missing_cost_countries.csv'),
    }
    if all(os.path.exists(v) for v in outputs.values()):
        p.L.info("Daily recreation flow outputs already exist, skipping calculation")
        return

    inputs = {
        'hq_sites': os.path.join(p.recreation_site_analysis_dir, 'high_quality_sites_1km.tif'),
        'hq_sites_dissolved': os.path.join(p.recreation_site_analysis_dir, 'hq_sites_dissolved_1km.tif'),
        'population': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/worldpop/ppp_2010_1km_aligned.tif'),
        'group_id': os.path.join(p.fit_recreation_demand_params_dir, 'group_id_1km.tif'),
        'country_id': os.path.join(p.allocate_overnights_dir, 'iso3_r250_ids_1km.tif'),
        'country_costs': os.path.join(p.data_dir, 'gep_recreation/0_inputs/feul_cost_per_km_2019_2datasources_iso3.csv'),
    }
    missing_files = [f"{k}: {v}" for k, v in inputs.items() if not os.path.exists(v)]
    if missing_files:
        p.L.error("Missing input files:\n" + "\n".join(missing_files))
        raise FileNotFoundError("Missing required input files")

    p.L.info("Starting daily recreation flow calculation")
    calculate_recreation_flows(p, inputs, outputs, _load_group_params(p), max_distance_km=50.0, max_bands=8)


def calculate_tourist_recreation_into_sites_kernel(p):
    """Overnight-visitor population -> site flows (visits, value, unmet demand)."""
    outputs = {
        'visits': os.path.join(p.cur_dir, 'site_visits_1km.tif'),
        'value': os.path.join(p.cur_dir, 'site_value_1km.tif'),
        'unmet': os.path.join(p.cur_dir, 'unmet_demand_1km.tif'),
        'site_table': os.path.join(p.cur_dir, 'site_visits_value.csv'),
        'group_table': os.path.join(p.cur_dir, 'param_group_qa.csv'),
        'missing_cost_iso3': os.path.join(p.cur_dir, 'missing_cost_countries.csv'),
    }
    if all(os.path.exists(v) for v in outputs.values()):
        p.L.info("Tourist recreation flow outputs already exist, skipping calculation")
        return

    inputs = {
        'hq_sites': os.path.join(p.recreation_site_analysis_dir, 'high_quality_sites_1km.tif'),
        'hq_sites_dissolved': os.path.join(p.recreation_site_analysis_dir, 'hq_sites_dissolved_1km.tif'),
        'population': os.path.join(p.allocate_overnights_dir, 'overnights_1km.tif'),
        'group_id': os.path.join(p.fit_recreation_demand_params_dir, 'group_id_1km.tif'),
        'country_id': os.path.join(p.allocate_overnights_dir, 'iso3_r250_ids_1km.tif'),
        'country_costs': os.path.join(p.data_dir, 'gep_recreation/0_inputs/feul_cost_per_km_2019_2datasources_iso3.csv'),
    }
    missing_files = [f"{k}: {v}" for k, v in inputs.items() if not os.path.exists(v)]
    if missing_files:
        p.L.error("Missing input files:\n" + "\n".join(missing_files))
        raise FileNotFoundError("Missing required input files")

    p.L.info("Starting tourist recreation flow calculation")
    calculate_recreation_flows(p, inputs, outputs, _load_group_params(p), max_distance_km=50.0, max_bands=8)


def aggregate_results_iso3(p):
    """Aggregate daily and tourist recreation results to country level,
    reading the site-level CSVs the flow calculation writes directly."""
    inputs = {
        'daily_site_table': os.path.join(p.calculate_daily_recreation_into_sites_kernel_dir, 'site_visits_value.csv'),
        'tourist_site_table': os.path.join(p.calculate_tourist_recreation_into_sites_kernel_dir, 'site_visits_value.csv'),
        'daily_missing_cost': os.path.join(p.calculate_daily_recreation_into_sites_kernel_dir, 'missing_cost_countries.csv'),
        'tourist_missing_cost': os.path.join(p.calculate_tourist_recreation_into_sites_kernel_dir, 'missing_cost_countries.csv'),
        'missing_overnights': os.path.join(p.allocate_overnights_dir, 'missing_overnights_iso3.csv'),
        'country_gpkg': os.path.join(p.data_dir, 'gep_recreation', '0_inputs', 'ee_r250_correspondence.gpkg'),
    }
    outputs = {
        'results_by_country_csv': os.path.join(p.cur_dir, 'results_by_country.csv'),
        'results_by_country_gpkg': os.path.join(p.cur_dir, 'results_by_country.gpkg'),
    }
    if all(os.path.exists(v) for v in outputs.values()):
        p.L.info("Aggregated results already exist, skipping aggregation")
        return

    missing_files = [f"{k}: {v}" for k, v in inputs.items() if not os.path.exists(v)]
    if missing_files:
        p.L.error("Missing input files for aggregate_results_iso3:\n" + "\n".join(missing_files))
        raise FileNotFoundError("Missing required input files for aggregate_results_iso3")

    daily = pd.read_csv(inputs['daily_site_table']).groupby('iso3_r250_id').agg(
        daily_visits=('visits', 'sum'), daily_value=('value', 'sum')).reset_index()
    tourist = pd.read_csv(inputs['tourist_site_table']).groupby('iso3_r250_id').agg(
        tourist_visits=('visits', 'sum'), tourist_value=('value', 'sum')).reset_index()

    df = daily.merge(tourist, on='iso3_r250_id', how='outer')
    for col in ['daily_visits', 'daily_value', 'tourist_visits', 'tourist_value']:
        if col not in df.columns:
            df[col] = 0.0
    df = df.fillna(0.0)

    # Ring-band convolution round-trips accumulate float32 noise that can
    # leave a true-zero flow as a tiny negative (~1e-9 scale seen in
    # practice) rather than exactly 0; clamp rather than leave a sign that
    # would misread as "negative demand" downstream.
    for col in ['daily_visits', 'daily_value', 'tourist_visits', 'tourist_value']:
        df[col] = df[col].clip(lower=0.0)

    # Countries with no fuel-cost data get cost_per_km=0 inside the flow
    # calculation, which reads as "value=0" (travel is free) rather than
    # "value is unknown". Overwrite value -- but not visits, which don't
    # depend on cost -- with NaN for those countries so a real zero isn't
    # confused with missing data downstream.
    missing_cost_daily = pd.read_csv(inputs['daily_missing_cost'])['iso3_r250_id']
    missing_cost_tourist = pd.read_csv(inputs['tourist_missing_cost'])['iso3_r250_id']
    df.loc[df['iso3_r250_id'].isin(missing_cost_daily), 'daily_value'] = np.nan
    df.loc[df['iso3_r250_id'].isin(missing_cost_tourist), 'tourist_value'] = np.nan

    # Countries with NO UNWTO Overnights data in any year (e.g. Indonesia,
    # which only reports Arrivals) get an all-zero overnights raster, so
    # BOTH tourist_visits and tourist_value are unknown there, not just
    # value -- unlike the cost gap above, the whole tourist-population
    # input is missing, not one factor in the value formula.
    missing_overnights = pd.read_csv(inputs['missing_overnights'])['iso3_r250_id']
    df.loc[df['iso3_r250_id'].isin(missing_overnights), ['tourist_visits', 'tourist_value']] = np.nan

    gdf = gpd.read_file(inputs['country_gpkg'])
    gdf_merged = gdf.merge(df, on='iso3_r250_id', how='left')
    gdf_out = gdf_merged[['iso3_r250_id', 'iso3_r250_label', 'daily_visits', 'daily_value',
                           'tourist_visits', 'tourist_value', 'geometry']]

    pd.DataFrame(gdf_out).drop(columns='geometry').to_csv(outputs['results_by_country_csv'], index=False)
    p.L.info(f"Saved country results to {outputs['results_by_country_csv']}")
    gdf_out.to_file(outputs['results_by_country_gpkg'], driver='GPKG')
    p.L.info(f"Saved country results to {outputs['results_by_country_gpkg']}")
