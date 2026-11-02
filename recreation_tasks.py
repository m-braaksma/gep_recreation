"""
recreation_tasks.py

ProjectFlow task functions for the recreation ecosystem service. Each task
resolves its input/output paths, skips if outputs already exist, and calls
into recreation_functions.py (science) and recreation_utils.py (helpers).
"""

import os
import shutil
import pickle
import pandas as pd
import geopandas as gpd
import pygeoprocessing

import recreation_functions as rf
import recreation_utils as ru


def _input_path(p, *parts):
    return os.path.join(p.data_dir, 'gep_recreation', *parts)


# ===========================================================================
# Recreation sites
# ===========================================================================

def generate_environment_index(p):
    lulc_dir = _input_path(p, '0_processed_raster_inputs', 'lulc')
    inputs = {k: os.path.join(lulc_dir, f'lulc_esa_seals7_2010_1km_{k}_share.tif')
              for k in ['cropland', 'forest', 'grassland', 'othernat', 'urban', 'water']}
    inputs['pa'] = _input_path(p, '0_processed_raster_inputs', 'pa', 'wdpa_accessible_1km_share.tif')
    output_path = os.path.join(p.cur_dir, 'env_index_1km.tif')
    if os.path.exists(output_path):
        p.L.info(f"Environment index already exists: {output_path}")
        return

    ru.require_files(inputs, 'generate_environment_index')
    p.L.info(f"Calculating environment index and saving to: {output_path}")
    rf.environment_index(inputs, output_path)


def generate_accessibility_index(p):
    inputs = {
        'urban': _input_path(p, '0_processed_raster_inputs', 'lulc', 'lulc_esa_seals7_2010_1km_urban_share.tif'),
        'road': _input_path(p, '0_processed_raster_inputs', 'distance_to_roads', 'distance_to_road_1km_corrected.tif'),
    }
    output_path = os.path.join(p.cur_dir, 'accessibility_index_1km.tif')
    if os.path.exists(output_path):
        p.L.info(f"Accessibility index already exists: {output_path}")
        return

    ru.require_files(inputs, 'generate_accessibility_index')
    p.L.info(f"Calculating accessibility index and saving to: {output_path}")
    rf.accessibility_index(inputs['urban'], inputs['road'], output_path)


def recreation_site_analysis(p):
    """
    1) Combine environment and accessibility indices to rank sites.
    2) Keep high-quality (rank 9) sites.
    3) Dissolve contiguous high-quality pixels into discrete sites.
    """
    env_path = os.path.join(p.generate_environment_index_dir, 'env_index_1km.tif')
    acc_path = os.path.join(p.generate_accessibility_index_dir, 'accessibility_index_1km.tif')
    ranked_path = os.path.join(p.cur_dir, 'rec_sites_ranked_1km.tif')
    hq_path = os.path.join(p.cur_dir, 'high_quality_sites_1km.tif')
    dissolved_path = os.path.join(p.cur_dir, 'hq_sites_dissolved_1km.tif')
    site_sizes_path = os.path.join(p.cur_dir, 'hq_site_pixel_counts.csv')

    if not os.path.exists(ranked_path):
        ru.require_files({'env_index': env_path, 'accessibility_index': acc_path}, 'rank_sites')
        p.L.info(f"Ranking recreation sites: {ranked_path}")
        rf.rank_sites(acc_path, env_path, ranked_path)

    if not os.path.exists(hq_path):
        p.L.info(f"Identifying high-quality recreation sites: {hq_path}")
        rf.high_quality_sites(ranked_path, hq_path)

    if not os.path.exists(dissolved_path):
        p.L.info("Dissolving contiguous high-quality-site pixels into discrete sites")
        rf.dissolve_sites(hq_path, dissolved_path, site_sizes_path)
        p.L.info(f"Saved dissolved sites to {dissolved_path} and pixel counts to {site_sizes_path}")


# ===========================================================================
# Demand-side inputs
# ===========================================================================

def allocate_overnights(p):
    """National UNWTO overnight stays spread across hotel locations."""
    workbook_path = _input_path(p, '0_inputs', 'unwto-all-data-download_2022.xlsx')
    country_gpkg_path = _input_path(p, '0_inputs', 'ee_r250_correspondence.gpkg')
    hotel_points_path = _input_path(p, '0_inputs', 'Export_GEO-data_4-University-of-Minnesota.gpkg')
    ref_raster_path = _input_path(p, '0_processed_raster_inputs', 'grip4_road_length', 'grip4_road_length_1km.tif')

    hotel_raster_path = os.path.join(p.cur_dir, 'hotels_1km.tif')
    country_raster_path = os.path.join(p.cur_dir, 'iso3_r250_ids_1km.tif')
    output_path = os.path.join(p.cur_dir, 'overnights_1km.tif')
    if os.path.exists(output_path):
        p.L.info(f"Overnight raster already exists: {output_path}")
        return

    ru.require_files({'unwto_workbook': workbook_path, 'country_gpkg': country_gpkg_path,
                      'hotel_points': hotel_points_path, 'reference_raster': ref_raster_path},
                     'allocate_overnights')

    p.L.info("Rasterizing hotel points...")
    ru.rasterize_points_count(hotel_points_path, ref_raster_path, hotel_raster_path)
    p.L.info("Rasterizing country borders...")
    ru.rasterize_attribute(country_gpkg_path, ref_raster_path, country_raster_path, 'iso3_r250_id')
    ru.check_country_raster(country_raster_path)

    p.L.info("Importing UNWTO overnight data...")
    overnight_df = rf.clean_unwto_overnights(workbook_path)
    overnight_df.to_csv(os.path.join(p.cur_dir, 'unwto_panel.csv'), index=False)

    country_overnights_map, substitution_report = rf.country_overnights(overnight_df)
    substitution_report.to_csv(os.path.join(p.cur_dir, 'overnights_year_substitution.csv'), index=False)
    _, uncovered = ru.log_country_coverage(
        country_raster_path, country_overnights_map.keys(), 'UNWTO overnights',
        'tourist demand there reads as a real 0 -- flagged for NaN downstream')
    pd.DataFrame({'iso3_r250_id': sorted(uncovered)}).to_csv(
        os.path.join(p.cur_dir, 'missing_overnights_iso3.csv'), index=False)

    p.L.info("Allocating overnights to hotel pixels...")
    rf.allocate_overnights_to_hotels(country_overnights_map, hotel_raster_path, country_raster_path, output_path)
    p.L.info(f"Overnight allocation complete: {output_path}")


def fit_recreation_demand_params(p):
    """Load + validate the external parameter table and build the aligned
    param_group id raster. Shared by both daily and tourist flows."""
    output_group_id_raster_path = os.path.join(p.cur_dir, 'group_id_1km.tif')
    if os.path.exists(output_group_id_raster_path):
        p.L.info("Parameter group raster already exists, skipping")
        return

    country_id_raster_path = os.path.join(p.allocate_overnights_dir, 'iso3_r250_ids_1km.tif')
    param_table_path = _input_path(p, '0_inputs', 'recreation_demand_params_global.csv')
    ru.require_files({'country_id': country_id_raster_path, 'param_table': param_table_path},
                     'fit_recreation_demand_params')

    country_to_group_map, group_params = rf.load_recreation_params(param_table_path)
    with open(os.path.join(p.cur_dir, 'group_params.pkl'), 'wb') as f:
        pickle.dump(group_params, f)

    rf.build_group_id_raster(country_id_raster_path, country_to_group_map, output_group_id_raster_path)
    p.L.info(f"Saved param_group id raster to {output_group_id_raster_path}")


def build_airfare_region_matrix(p):
    """
    Copies the pinned regional airfare survey (0_inputs/airfare_2019_regions.csv)
    into the project. The survey itself is a live scrape and is rebuilt only
    deliberately, by running airfare_survey_scrape.py, so reruns of the
    pipeline never silently change air-travel values.
    """
    static_path = _input_path(p, '0_inputs', 'airfare_2019_regions.csv')
    output_path = os.path.join(p.cur_dir, 'airfare_2019_regions.csv')
    if os.path.exists(output_path):
        p.L.info(f"Airfare region matrix already exists: {output_path}")
        return
    if not os.path.exists(static_path):
        raise FileNotFoundError(
            f"Missing pinned airfare survey: {static_path}. Run airfare_survey_scrape.py "
            "to build it (live network requests; results depend on the survey date).")
    p.L.info(f"Using pinned airfare region matrix: {static_path}")
    shutil.copy(static_path, output_path)


# ===========================================================================
# Flows
# ===========================================================================

def _load_group_params(p):
    with open(os.path.join(p.fit_recreation_demand_params_dir, 'group_params.pkl'), 'rb') as f:
        return pickle.load(f)


def _run_flows(p, population_path, group_params, label, population_scale=1.0):
    outputs = {
        'visits': os.path.join(p.cur_dir, 'site_visits_1km.tif'),
        'value': os.path.join(p.cur_dir, 'site_value_1km.tif'),
        'unmet': os.path.join(p.cur_dir, 'unmet_demand_1km.tif'),
        'site_table': os.path.join(p.cur_dir, 'site_visits_value.csv'),
        'group_table': os.path.join(p.cur_dir, 'param_group_qa.csv'),
        'missing_cost_iso3': os.path.join(p.cur_dir, 'missing_cost_countries.csv'),
    }
    if ru.outputs_exist(outputs):
        p.L.info(f"{label} flow outputs already exist, skipping calculation")
        return

    inputs = {
        'hq_sites': os.path.join(p.recreation_site_analysis_dir, 'high_quality_sites_1km.tif'),
        'hq_sites_dissolved': os.path.join(p.recreation_site_analysis_dir, 'hq_sites_dissolved_1km.tif'),
        'population': population_path,
        'group_id': os.path.join(p.fit_recreation_demand_params_dir, 'group_id_1km.tif'),
        'country_id': os.path.join(p.allocate_overnights_dir, 'iso3_r250_ids_1km.tif'),
        'country_costs': _input_path(p, '0_inputs', 'feul_cost_per_km_2019_2datasources_iso3.csv'),
    }
    ru.require_files(inputs, f'{label} flows')
    inputs['population_scale'] = population_scale

    p.L.info(f"Starting {label} flow calculation")
    rf.calculate_recreation_flows(inputs, outputs, group_params, work_dir=os.path.join(p.cur_dir, 'flows'))


def calculate_daily_recreation_into_sites_kernel(p):
    """Resident population -> site flows (visits, value, unmet demand)."""
    population_path = _input_path(p, '0_processed_raster_inputs', 'worldpop', 'ppp_2020_1km_aligned.tif')
    _run_flows(p, population_path, _load_group_params(p), 'Daily recreation',
               population_scale=rf.WORLDPOP_AREA_SCALE)


def calculate_tourist_recreation_into_sites_kernel(p):
    """Overnight-visitor population -> site flows (visits, value, unmet demand)."""
    # The tourist population is person-nights per year, but the participation
    # rate a is visits per person per year. Scaling a to a per-night rate keeps
    # a tourist's visits proportional to nights stayed (a seven-night stay gives
    # about two visits, not 7 x a).
    group_params = {gid: {'a': v['a'] / 365.0, 'b': v['b']} for gid, v in _load_group_params(p).items()}
    population_path = os.path.join(p.allocate_overnights_dir, 'overnights_1km.tif')
    _run_flows(p, population_path, group_params, 'Tourist recreation')


# ===========================================================================
# Valuation
# ===========================================================================

def calculate_accommodation_value(p):
    """
    Values overnight stays at the hotel prices they were spent at, using the
    pre-built gap-filled hotel-price surface as-is (observed pixel -> kriged
    -> nearest-observed-within-500km -> country mean; see
    README_Interpolation.md alongside the source rasters).
    """
    price_path = _input_path(p, '0_inputs', 'booking', 'hotel_price_per_occupancy_mean_0p5deg_filled.tif')
    overnights_path = os.path.join(p.allocate_overnights_dir, 'overnights_1km.tif')
    country_id_path = os.path.join(p.allocate_overnights_dir, 'iso3_r250_ids_1km.tif')

    price_1km_path = os.path.join(p.cur_dir, 'hotel_price_per_occupancy_mean_1km.tif')
    value_path = os.path.join(p.cur_dir, 'accommodation_value_1km.tif')
    country_table_path = os.path.join(p.cur_dir, 'accommodation_value_by_country.csv')
    if ru.outputs_exist([value_path, country_table_path]):
        p.L.info("Accommodation value already exists, skipping calculation")
        return

    ru.require_files({'hotel_price': price_path, 'overnights': overnights_path, 'country_id': country_id_path},
                     'calculate_accommodation_value')

    if not os.path.exists(price_1km_path):
        p.L.info(f"Resampling 0.5-degree hotel price surface onto the 1km grid: {price_1km_path}")
        overnights_info = pygeoprocessing.get_raster_info(overnights_path)
        pygeoprocessing.warp_raster(
            price_path, overnights_info['pixel_size'], price_1km_path, 'bilinear',
            target_bb=overnights_info['bounding_box'],
            target_projection_wkt=overnights_info['projection_wkt'])

    p.L.info(f"Calculating accommodation value and saving to: {value_path}")
    rf.accommodation_value_raster(overnights_path, price_1km_path, value_path)

    p.L.info("Aggregating accommodation value and overnights price-coverage by country")
    country_df = rf.accommodation_value_by_country(overnights_path, country_id_path, value_path)
    country_df.to_csv(country_table_path, index=False)
    p.L.info(f"Saved accommodation value + price-coverage table to {country_table_path}")
    low_coverage = country_df[country_df['price_coverage_share'] < 0.5]
    if not low_coverage.empty:
        p.L.warning(f"{len(low_coverage)} countries have hotel-price coverage under 50% of their overnight-stays "
                    f"(accommodation_value for these is a partial sum, not a full-country estimate): "
                    f"{sorted(low_coverage['iso3_r250_id'].tolist())}")


def calculate_air_travel_value(p):
    """
    Values international air travel to reach the destination country:
    arrivals by origin region x air share x regional airfare. Also writes
    the personal-purpose share of arrivals used in aggregation.
    """
    workbook_path = _input_path(p, '0_inputs', 'unwto-all-data-download_2022.xlsx')
    airfare_path = os.path.join(p.build_airfare_region_matrix_dir, 'airfare_2019_regions.csv')
    crosswalk_path = _input_path(p, '0_inputs', 'unwto_region_crosswalk.csv')
    cpi_path = _input_path(p, '0_inputs', 'CUUR0000SETG01.csv')

    output_path = os.path.join(p.cur_dir, 'air_travel_value_by_country.csv')
    personal_share_path = os.path.join(p.cur_dir, 'personal_share_by_country.csv')
    if ru.outputs_exist([output_path, personal_share_path]):
        p.L.info(f"Air travel value already exists: {output_path}")
        return

    ru.require_files({'arrivals_workbook': workbook_path, 'airfare_matrix': airfare_path,
                      'region_crosswalk': crosswalk_path, 'airfare_cpi': cpi_path},
                     'calculate_air_travel_value')

    p.L.info("Cleaning UNWTO arrivals-by-origin-region data")
    arrivals_chosen = rf.select_arrivals_year(rf.clean_unwto_arrivals_by_region(workbook_path))

    airfare = pd.read_csv(airfare_path)
    if 'survey_query_date' not in airfare.columns:
        raise ValueError(f"{airfare_path} has no survey_query_date column; needed to deflate fares to {rf.TARGET_YEAR}.")
    survey_date = airfare['survey_query_date'].iloc[0]
    deflation_factor, cpi_month = rf.airfare_deflation_factor(cpi_path, survey_date)
    airfare['predicted_fare_2019_usd'] = airfare['average_fare_usd_current'] * deflation_factor
    p.L.info(f"Deflating fares surveyed {survey_date} to {rf.TARGET_YEAR} with airline-fare CPI "
             f"(month used: {cpi_month.date()}): factor = {deflation_factor:.4f}")
    airfare = airfare[['origin_region', 'destination_region', 'predicted_fare_2019_usd']]

    air_share = rf.clean_unwto_air_share(workbook_path, crosswalk_path)
    air_share.to_csv(os.path.join(p.cur_dir, 'air_share_by_country.csv'), index=False)
    p.L.info(f"Air share of arrivals: measured for {int((~air_share['air_share_imputed']).sum())} countries, "
             f"regional-median imputed for {int(air_share['air_share_imputed'].sum())}")

    personal_share = rf.clean_unwto_personal_share(workbook_path, crosswalk_path)
    personal_share.to_csv(personal_share_path, index=False)
    p.L.info(f"Personal-purpose share of arrivals: measured for "
             f"{int((~personal_share['personal_share_imputed']).sum())} countries, regional-median imputed for "
             f"{int(personal_share['personal_share_imputed'].sum())} (median "
             f"{personal_share['personal_share'].median():.3f})")

    country_df = rf.air_travel_value_by_country(arrivals_chosen, crosswalk_path, airfare, air_share)
    country_df.to_csv(output_path, index=False)
    p.L.info(f"Saved air travel value table to {output_path}")
    low_coverage = country_df[country_df['arrivals_coverage_share'] < 0.5]
    if not low_coverage.empty:
        p.L.warning(f"{len(low_coverage)} countries have arrivals priced for under 50% of their total "
                    f"(air_travel_value for these is a partial sum): {sorted(low_coverage['iso3_r250_id'].tolist())}")


# ===========================================================================
# Aggregation
# ===========================================================================

def aggregate_results_iso3(p):
    """Combine the four value components to country level."""
    daily_dir = p.calculate_daily_recreation_into_sites_kernel_dir
    tourist_dir = p.calculate_tourist_recreation_into_sites_kernel_dir
    inputs = {
        'daily_site_table': os.path.join(daily_dir, 'site_visits_value.csv'),
        'tourist_site_table': os.path.join(tourist_dir, 'site_visits_value.csv'),
        'daily_missing_cost': os.path.join(daily_dir, 'missing_cost_countries.csv'),
        'tourist_missing_cost': os.path.join(tourist_dir, 'missing_cost_countries.csv'),
        'missing_overnights': os.path.join(p.allocate_overnights_dir, 'missing_overnights_iso3.csv'),
        'accommodation_table': os.path.join(p.calculate_accommodation_value_dir, 'accommodation_value_by_country.csv'),
        'air_travel_table': os.path.join(p.calculate_air_travel_value_dir, 'air_travel_value_by_country.csv'),
        'personal_share': os.path.join(p.calculate_air_travel_value_dir, 'personal_share_by_country.csv'),
        'country_gpkg': _input_path(p, '0_inputs', 'ee_r250_correspondence.gpkg'),
    }
    outputs = {
        'results_by_country_csv': os.path.join(p.cur_dir, 'results_by_country.csv'),
        'results_by_country_gpkg': os.path.join(p.cur_dir, 'results_by_country.gpkg'),
        'gep_results_csv': os.path.join(p.cur_dir, 'gep_recreation_results.csv'),
    }
    if ru.outputs_exist(outputs):
        p.L.info("Aggregated results already exist, skipping aggregation")
        return
    ru.require_files(inputs, 'aggregate_results_iso3')

    gdf = gpd.read_file(inputs['country_gpkg'])
    df = rf.combine_country_results(
        country_ids=gdf['iso3_r250_id'],
        daily_sites=pd.read_csv(inputs['daily_site_table']),
        tourist_sites=pd.read_csv(inputs['tourist_site_table']),
        missing_cost_daily=pd.read_csv(inputs['daily_missing_cost'])['iso3_r250_id'],
        missing_cost_tourist=pd.read_csv(inputs['tourist_missing_cost'])['iso3_r250_id'],
        missing_overnights=pd.read_csv(inputs['missing_overnights'])['iso3_r250_id'],
        accommodation=pd.read_csv(inputs['accommodation_table']),
        air_travel=pd.read_csv(inputs['air_travel_table']),
        personal_share=pd.read_csv(inputs['personal_share']),
    )

    gdf_out = gdf.merge(df, on='iso3_r250_id', how='left')[[
        'iso3_r250_id', 'iso3_r250_label', 'daily_visits', 'daily_value',
        'tourist_visits', 'tourist_value', 'accommodation_value',
        'accommodation_value_pre_attribution', 'price_coverage_share', 'air_travel_value',
        'air_travel_value_pre_attribution', 'arrivals_coverage_share', 'personal_share',
        'personal_share_imputed', 'grand_total_value', 'n_components_missing', 'geometry']]

    pd.DataFrame(gdf_out).drop(columns='geometry').to_csv(outputs['results_by_country_csv'], index=False)
    p.L.info(f"Saved country results to {outputs['results_by_country_csv']}")
    gdf_out.to_file(outputs['results_by_country_gpkg'], driver='GPKG')
    p.L.info(f"Saved country results to {outputs['results_by_country_gpkg']}")
    ru.format_gep_table(df).to_csv(outputs['gep_results_csv'], index=False, encoding='utf-8-sig')
    p.L.info(f"Saved GEP results table to {outputs['gep_results_csv']}")
