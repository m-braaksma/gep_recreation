import numpy as np
import pygeoprocessing
from osgeo import gdal
import os
import pandas as pd
import geopandas as gpd

# Reset GDAL_DATA path after importing hazelbean
gdal_data_path = os.environ.get("GDAL_DATA")
print("GDAL_DATA:", gdal_data_path)
os.environ['GDAL_DATA'] = '/Users/mbraaksma/mambaforge/envs/hb01/share/gdal'

def generate_environment_index(p):
    """
    Generate environment index raster for recreation site analysis.
    
    Combines land use/land cover data with protected area information
    to create a composite environmental quality index.
    
    Args:
        p: Parameters object containing data_dir, cur_dir, and logger
        
    Returns:
        str: Path to generated environment index raster
        
    Raises:
        FileNotFoundError: If required input files are missing
    """

    # Define input paths relative to data_dir
    inputs = {
        'cropland': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/lulc/lulc_esa_seals7_2010_1km_cropland_share.tif'),
        'forest': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/lulc/lulc_esa_seals7_2010_1km_forest_share.tif'),
        'grassland': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/lulc/lulc_esa_seals7_2010_1km_grassland_share.tif'),
        'othernat': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/lulc/lulc_esa_seals7_2010_1km_othernat_share.tif'),
        'urban': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/lulc/lulc_esa_seals7_2010_1km_urban_share.tif'),
        'water': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/lulc/lulc_esa_seals7_2010_1km_water_share.tif'),
        'pa': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/pa/wdpa_accessible_1km_share.tif'),
    }
    
    
    # Define output path
    output_path = os.path.join(p.cur_dir, 'env_index_1km.tif')
    if os.path.exists(output_path):
        p.L.info(f"Environment index already exists: {output_path}")
        return

    lulc_scores = {
        'cropland': 0.4,
        'forest': 1.0,
        'grassland': 0.7,
        'othernat': 0.85,
        'water': 1.0,
        'urban': 0.05,
        'other': 0.0
    }
    pa_score = 1.0
    env_classify = {
        'high': 1.5,
        'med_high': 1.0,
        'med_low': 0.5,
        'low': 0
    }
    nodata_value = -1

    def environment_index_op(crop, forest, grass, othernat, urban, water, pa):
        """Raster calculator function for recreation potential"""
        # Weighted LULC composite
        lulc_potential = (
            lulc_scores['cropland'] * crop +
            lulc_scores['forest'] * forest +
            lulc_scores['grassland'] * grass +
            lulc_scores['othernat'] * othernat +
            lulc_scores['urban'] * urban +
            lulc_scores['water'] * water
        )
        # Combined score
        combined = lulc_potential + pa_score * pa
        # Classify into bins (0–3)
        env_class = np.select(
            condlist=[
                combined > env_classify['high'],
                combined > env_classify['med_high'],
                combined > env_classify['med_low'],
                combined >= env_classify['low']
            ],
            choicelist=[3, 2, 1, 0],
            default=0
        )
        return env_class
    
    # Check if all input files exist
    missing_files = []
    for key, path in inputs.items():
        if not os.path.exists(path):
            missing_files.append(f"{key}: {path}")
    if missing_files:
        p.L.warning(f"Missing input files for generate_environment_index:\n" + "\n".join(missing_files))
        raise FileNotFoundError(f"Missing input files for generate_environment_index:\n" + "\n".join(missing_files))

    # Prepare band list for calculator
    base_list = [
        (inputs['cropland'], 1),
        (inputs['forest'], 1),
        (inputs['grassland'], 1),
        (inputs['othernat'], 1),
        (inputs['urban'], 1),
        (inputs['water'], 1),
        (inputs['pa'], 1),
    ]
    p.L.info(f"Calculating environment index and saving to: {output_path}")

    # Run raster calculator
    pygeoprocessing.raster_calculator(
        base_list,
        environment_index_op,
        output_path,
        gdal.GDT_Int32,
        nodata_value,
        calc_raster_stats=True,
    )
    p.L.info(f"Environment index calculation completed: {output_path}")


def generate_accessibility_index(p):
    """
    Generate accessibility index raster for recreation site analysis.
    
    Combines urban land use/land cover data with road data
    to create a composite accessibility index.
    
    Args:
        p: Parameters object containing data_dir, cur_dir, and logger
        
    Returns:
        str: Path to generated accessibility index raster
        
    Raises:
        FileNotFoundError: If required input files are missing
    """
    # Define input paths relative to data_dir
    inputs = {
        'urban': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/lulc/lulc_esa_seals7_2010_1km_urban_share.tif'),
        'road': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/distance_to_roads/distance_to_road_1km_corrected.tif')
    }
    
    # Define output path
    output_path = os.path.join(p.cur_dir, 'accessibility_index_1km.tif')
    if os.path.exists(output_path):
        p.L.info(f"Accessibility index already exists: {output_path}")
        return
    
    # Configuration parameters
    urban_classify = {
        'high': 0.9,
        'med_high': 0.5,
        'med': 0.1,
        'med_low': 0.001,
        'low': 0
    }
    road_classify = {
        'high': 50,
        'med_high': 20,
        'med': 5,
        'med_low': 2,
        'low': 0
    }
    urban_road_matrix = np.array([
        [1, 1, 2, 3, 4],
        [1, 1, 2, 3, 4],
        [2, 2, 2, 4, 5],
        [3, 3, 4, 5, 5],
        [3, 4, 4, 5, 5]
    ])
    nodata_value = -1
    def accessibility_op(urban, roads):
        """Raster calculator function for accessibility index"""
        urban_class = np.select(
            condlist=[
                urban > urban_classify['high'],
                urban > urban_classify['med_high'],
                urban > urban_classify['med'],
                urban > urban_classify['med_low'],
                urban >= urban_classify['low']
            ],
            choicelist=[4, 3, 2, 1, 0], 
            default=0
        )
        road_class = np.select(
            condlist=[
                roads > road_classify['high'],
                roads > road_classify['med_high'],
                roads > road_classify['med'],
                roads > road_classify['med_low'],
                roads >= road_classify['low']
            ],
            choicelist=[0, 1, 2, 3, 4],
            default=0
        )
        return urban_road_matrix[urban_class, road_class].astype(np.int32)
    
    # Check if all input files exist
    missing_files = []
    for key, path in inputs.items():
        if not os.path.exists(path):
            missing_files.append(f"{key}: {path}")
    if missing_files:
        p.L.warning(f"Missing input files for generate_accessibility_index:\n" + "\n".join(missing_files))
        raise FileNotFoundError(f"Missing input files for generate_accessibility_index:\n" + "\n".join(missing_files))

    # Prepare band list for calculator
    base_list = [
        (inputs['urban'], 1),
        (inputs['road'], 1),
    ]
    p.L.info(f"Calculating accessibility index (human inputs) and saving to: {output_path}")

    # Run raster calculator
    pygeoprocessing.raster_calculator(
        base_list,
        accessibility_op,
        output_path,
        gdal.GDT_Int32,
        nodata_value,
    )
    p.L.info(f"Accessibility index calculation completed: {output_path}")


def rank_rec_sites(p):
    # Define input paths from previous tasks
    inputs = {
        'env_index': os.path.join(p.generate_environment_index_dir, 'env_index_1km.tif'),
        'accessibility_index': os.path.join(p.generate_accessibility_index_dir, 'accessibility_index_1km.tif')
    }
    
    # Define output path
    output_path = os.path.join(p.cur_dir, 'rec_sites_ranked_1km.tif')
    if os.path.exists(output_path):
         p.L.info(f"Recreation site ranking already exists: {output_path}")
         return
        
    # Configuration parameters
    site_matrix = np.array([
        [1, 1, 4, 7],
        [1, 4, 4, 7],
        [2, 2, 8, 8],
        [3, 5, 5, 9],
        [3, 6, 6, 9]
    ])

    nodata_value = -1
    def site_op(acc_class, env_class):
        """Raster calculator function for ranking classification"""
        acc_class = acc_class - 1
        return site_matrix[acc_class, env_class].astype(np.int16)
    
    # Check if all input files exist
    missing_files = []
    for key, path in inputs.items():
        if not os.path.exists(path):
            missing_files.append(f"{key}: {path}")
    if missing_files:
        p.L.warning(f"Missing input files for rank_rec_sites:\n" + "\n".join(missing_files))
        raise FileNotFoundError("Missing input files for rank_rec_sites:\n" + "\n".join(missing_files))
    
    # Prepare band list for calculator
    base_list = [
        (inputs['accessibility_index'], 1),
        (inputs['env_index'], 1),
    ]
    p.L.info(f"Calculating site classification and saving to: {output_path}")

    # Run raster calculator
    pygeoprocessing.raster_calculator(
        base_list,
        site_op,
        output_path,
        gdal.GDT_Int32,
        nodata_value,
    )
    p.L.info(f"Recreation site classification completed: {output_path}")


def find_high_quality_rec_sites(p):
    # Define input paths from previous task
    inputs = {
        'ranked_sites': os.path.join(p.recreation_site_analysis_dir, 'rec_sites_ranked_1km.tif')
    }
    
    # Define output path
    output_path = os.path.join(p.cur_dir, 'high_quality_sites_1km.tif')
    if os.path.exists(output_path):
        p.L.info(f"High quality sites already identified: {output_path}")
        return
    
    def hq_op(site_class):
        """Raster calculator function for high quality sites"""
        return np.where(site_class == 9, 1, 0).astype(np.int32)
    
    # Check if input file exists
    if not os.path.exists(inputs['ranked_sites']):
        p.L.warning(f"Missing input file for find_high_quality_rec_sites: {inputs['ranked_sites']}")
        raise FileNotFoundError(f"Missing input file for find_high_quality_rec_sites: {inputs['ranked_sites']}")

    # Prepare band list for calculator
    base_list = [(inputs['ranked_sites'], 1)]
    p.L.info(f"Identifying high quality recreation sites and saving to: {output_path}")
    nodata_value = -1

    # Run raster calculator
    pygeoprocessing.raster_calculator(
        base_list,
        hq_op,
        output_path,
        gdal.GDT_Int32,
        nodata_value,
    )
    p.L.info(f"High quality sites identification completed: {output_path}")


def recreation_site_analysis(p):
    """
    1) Combine environment index and accessibility index to rank recreation sites.
    2) Identify high quality recreation sites. 
    3) Generate raster of distance to high quality sites.
    
    Args:
        p: Parameters object containing data_dir, cur_dir, and logger
        
    Returns:
        str: Output paths for ranked raster, high quality site raster, and distance raster
        
    Raises:
        FileNotFoundError: If required input files are missing
    """
    rank_rec_sites(p)
    find_high_quality_rec_sites(p)


def calculate_daily_recreation_into_sites_kernel(p):
    """
    Calculate daily recreation visits and values flowing INTO recreation sites
    using kernel-based convolution for better performance on large rasters.
    """
    # Define input paths
    inputs = {
        'hq_sites': os.path.join(p.recreation_site_analysis_dir, 'high_quality_sites_1km.tif'),
        'population': os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/worldpop/ppp_2010_1km_aligned.tif'),
        'country_id': os.path.join(p.allocate_overnights_dir, 'iso3_r250_ids_1km.tif'),
        'country_costs': os.path.join(p.data_dir, 'gep_recreation/0_inputs/feul_cost_per_km_2019_2datasources_iso3.csv')
    }
    
    # Define output paths
    outputs = {
        'visits': os.path.join(p.cur_dir, 'site_visits_1km.tif'),
        'value': os.path.join(p.cur_dir, 'site_value_1km.tif'),
    }
    
    # Check if outputs already exist
    if all(os.path.exists(output) for output in outputs.values()):
        p.L.info("Recreation site visit outputs already exist, skipping calculation")
        return
    
    # Check if all input files exist
    missing_files = []
    for key, path in inputs.items():
        if not os.path.exists(path):
            missing_files.append(f"{key}: {path}")
    
    if missing_files:
        p.L.error(f"Missing input files:\n" + "\n".join(missing_files))
        raise FileNotFoundError("Missing required input files")
    
    p.L.info("Starting kernel-based recreation site calculation")
    
    # Calculate for each distance buffer and sum results
    calculate_kernel_based_recreation(inputs, outputs, p)


def calculate_kernel_based_recreation(inputs, outputs, p):
    """
    Use kernel convolution to calculate recreation visits and values.
    """
    # Model parameters from your original function
    model_params = {
        'dist_buffers': [1, 2, 3, 4],  # km distances
        'k_vals': [0.0132, 0.0267, 0.0518, 0.1067],
        'alpha_vals': [0.00155, 0.00115, 0.00098, 0.00067],
        'weeks_per_year': 52
    }

    # Load country costs
    country_costs_df = pd.read_csv(inputs['country_costs'])
    country_costs = dict(zip(country_costs_df['iso3_r250_id'], 
                           country_costs_df['gasoline_cost_usd_per_km_2019_gppdata']))

    # Get raster info
    raster_info = pygeoprocessing.get_raster_info(inputs['population'])
    pixel_size = abs(raster_info['pixel_size'][0])  # Assuming square pixels
    
    # Initialize total arrays
    kernel_dir = os.path.join(p.cur_dir, 'kernel')
    os.makedirs(kernel_dir, exist_ok=True)
    total_visits_path = os.path.join(kernel_dir, 'total_visits.tif')
    total_value_path = os.path.join(kernel_dir, 'total_value.tif')
    
    # Initialize zero arrays for totals
    pygeoprocessing.new_raster_from_base(
        inputs['population'], total_visits_path, gdal.GDT_Float32, [-9999],
        fill_value_list=[0.0])
    pygeoprocessing.new_raster_from_base(
        inputs['population'], total_value_path, gdal.GDT_Float32, [-9999],
        fill_value_list=[0.0])
    
    # Process each distance buffer
    for buffer_idx, buffer_zone in enumerate(model_params['dist_buffers']):
        p.L.info(f"Processing buffer zone {buffer_zone} (grid cells)")
        
        # Create kernel for this buffer zone
        kernel_path = os.path.join(kernel_dir, f'kernel_buffer_{buffer_zone}.tif')
        create_distance_kernel(kernel_path, buffer_zone)
        
        # Create population visit potential surface for this buffer zone
        pop_potential_path = os.path.join(kernel_dir, f'pop_visit_potential_buffer_{buffer_zone}.tif')
        create_population_visit_potential_surface(
            inputs, pop_potential_path, 
            model_params, buffer_idx)
        
        # Convolve sites with population visit potential
        visits_buffer_path = os.path.join(kernel_dir, f'visits_buffer_{buffer_zone}.tif')
        value_buffer_path = os.path.join(kernel_dir, f'value_buffer_{buffer_zone}.tif')
        
        # Convolve for visits (population visit potential * sites)
        pygeoprocessing.convolve_2d(
            (pop_potential_path, 1),
            (kernel_path, 1),
            visits_buffer_path,
            ignore_nodata_and_edges=False,
            mask_nodata=True,
            normalize_kernel=False,
            target_datatype=gdal.GDT_Float32,
            target_nodata=-9999,
            working_dir=kernel_dir
        )
        
        # Values need cost-weighted visit potential
        value_potential_path = os.path.join(kernel_dir, f'pop_value_potential_buffer_{buffer_zone}.tif')
        create_value_potential_surface(
            inputs, value_potential_path, country_costs, 
            model_params, buffer_idx, buffer_zone, pixel_size)
        
        pygeoprocessing.convolve_2d(
            (value_potential_path, 1),
            (kernel_path, 1),
            value_buffer_path,
            ignore_nodata_and_edges=False,
            mask_nodata=True,
            normalize_kernel=False,
            target_datatype=gdal.GDT_Float32,
            target_nodata=-9999,
            working_dir=kernel_dir
        )
        
        # Add this buffer's contribution to totals
        add_to_total(total_visits_path, visits_buffer_path)
        add_to_total(total_value_path, value_buffer_path)
    
    # Mask results to only high-quality sites (measure only visits to high quality sites)
    mask_to_sites(total_visits_path, inputs['hq_sites'], outputs['visits'])
    mask_to_sites(total_value_path, inputs['hq_sites'], outputs['value'])
    
    p.L.info("Kernel-based recreation calculation completed")


def create_distance_kernel(kernel_path, buffer_zone):
    """
    Create a grid-based zonal kernel where buffer_zone determines the ring.
    Buffer 1 = all adjacent cells (including diagonals).
    Buffer 2 = cells 2 steps away in any direction, etc.
    """
    kernel_size = 2 * buffer_zone + 1
    kernel_array = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    center = buffer_zone

    # Use Chebyshev distance (max of row/col offset)
    for i in range(kernel_size):
        for j in range(kernel_size):
            chebyshev_dist = max(abs(i - center), abs(j - center))

            # Include center for first buffer
            if buffer_zone == 1: 
                if chebyshev_dist <= buffer_zone:
                    kernel_array[i, j] = 1.0
            else:
                if chebyshev_dist == buffer_zone:
                    kernel_array[i, j] = 1.0

    # Export to pygeoprocessing-compatible kernel
    n_dimensions = np.array(kernel_array).ndim
    if n_dimensions != 2:
        raise ValueError(
            "The kernel array must have exactly 2 dimensions, not "
            f"{n_dimensions}")
    pygeoprocessing.numpy_array_to_raster(
        kernel_array, target_nodata=None, pixel_size=None, origin=None,
        projection_wkt=None, target_path=kernel_path)
    

def create_population_visit_potential_surface(inputs, output_path, model_params, buffer_idx):
    """
    Create population visit potential surface using the gravity model formula.
    This represents "how many visits would this population cell generate 
    IF it happens to be in buffer zone X?"
    """
    k = model_params['k_vals'][buffer_idx]
    alpha = model_params['alpha_vals'][buffer_idx]
    weeks_per_year = model_params['weeks_per_year']
    
    def calculate_visit_potential_op(population, country_id):
        """Apply gravity model to calculate visit potential"""
        # Initialize output
        result = np.full_like(population, -9999, dtype=np.float32)
        
        # Find valid pixels
        valid_mask = (~np.isnan(population) & ~np.isnan(country_id) & 
                     (population > 0) & (country_id >= 0))
        
        if np.any(valid_mask):
            valid_pop = population[valid_mask]
            
            # Calculate weekly visits using gravity model
            weekly_visits = (1 + k) / (k + np.exp(-alpha * valid_pop))
            annual_visits = weekly_visits * weeks_per_year
            
            result[valid_mask] = annual_visits
            result[~valid_mask] = 0  # No visit potential from invalid pixels
        
        return result
    
    # Apply calculation
    pygeoprocessing.raster_calculator(
        [(inputs['population'], 1), (inputs['country_id'], 1)],
        calculate_visit_potential_op,
        output_path,
        gdal.GDT_Float32,
        -9999
    )


def create_value_potential_surface(inputs, output_path, country_costs, 
                                  model_params, buffer_idx, buffer_zone, pixel_size):
    """
    Create value potential surface (visits * travel cost).
    This represents "what economic value would this population cell generate 
    IF it happens to be in buffer zone X?"
    """
    k = model_params['k_vals'][buffer_idx]
    alpha = model_params['alpha_vals'][buffer_idx]
    weeks_per_year = model_params['weeks_per_year']
    
    # Create lookup array for costs
    max_country_id = max(country_costs.keys()) if country_costs else 0
    cost_lookup = np.zeros(max_country_id + 1, dtype=np.float32)
    for cid, cost in country_costs.items():
        cost_lookup[cid] = cost
    
    def calculate_value_potential(population, country_id):
        """Calculate value potential (visits * cost)"""
        result = np.full_like(population, -9999, dtype=np.float32)
        
        valid_mask = (~np.isnan(population) & ~np.isnan(country_id) & 
                     (population > 0) & (country_id >= 0) & 
                     (country_id <= max_country_id))
        
        if np.any(valid_mask):
            valid_pop = population[valid_mask]
            valid_countries = country_id[valid_mask].astype(int)
            
            # Calculate visits
            weekly_visits = (1 + k) / (k + np.exp(-alpha * valid_pop))
            annual_visits = weekly_visits * weeks_per_year
            
            # Calculate travel costs - convert buffer zone to approximate distance
            # Assuming each buffer zone represents some distance unit
            approx_distance = buffer_zone * pixel_size  # Convert to km if pixel_size is in km
            travel_costs = cost_lookup[valid_countries] * (approx_distance - 0.5 * pixel_size)
            
            # Value = visits * cost
            value_potential = annual_visits * travel_costs
            
            result[valid_mask] = value_potential
            result[~valid_mask] = 0
        
        return result
    
    pygeoprocessing.raster_calculator(
        [(inputs['population'], 1), (inputs['country_id'], 1)],
        calculate_value_potential,
        output_path,
        gdal.GDT_Float32,
        -9999
    )


def add_to_total(total_path, addition_path):
    """Add values from addition_path to total_path."""
    def add_arrays_op(total, addition):
        # Handle nodata
        valid_total = total != -9999
        valid_addition = addition != -9999
        
        result = np.full_like(total, -9999, dtype=np.float32)
        
        # Where both are valid, add them
        both_valid = valid_total & valid_addition
        result[both_valid] = total[both_valid] + addition[both_valid]
        
        # Where only total is valid, keep total
        only_total = valid_total & ~valid_addition
        result[only_total] = total[only_total]
        
        # Where only addition is valid, use addition
        only_addition = ~valid_total & valid_addition
        result[only_addition] = addition[only_addition]

        # Clean up floating-point precision errors - ensure no negative values
        valid_result = result != -9999
        result[valid_result] = np.maximum(result[valid_result], 0.0)
        
        return result
    
    temp_path = total_path.replace('.tif', '_temp.tif')
    pygeoprocessing.raster_calculator(
        [(total_path, 1), (addition_path, 1)],
        add_arrays_op,
        temp_path,
        gdal.GDT_Float32,
        -9999
    )
    
    # Replace original with sum
    os.replace(temp_path, total_path)


def mask_to_sites(input_path, sites_path, output_path):
    """Mask results to only show values at high-quality sites."""
    def apply_site_mask_op(values, sites):
        result = np.full_like(values, -9999, dtype=np.float32)
        site_mask = sites == 1
        result[site_mask] = values[site_mask]
        return result
    
    pygeoprocessing.raster_calculator(
        [(input_path, 1), (sites_path, 1)],
        apply_site_mask_op,
        output_path,
        gdal.GDT_Float32,
        -9999
    )



def allocate_overnights(p):
    """
    Allocate national overnight stays based on hotel points.
    1) Read overnight stay data from UNTWO excel file and clean.
    2) Map overnight data to country gkpg using ee_r250_id correspondence. 
    3) Rasterize to 1km grid. 
    4) Allocate to 1km grid based on hotel points.
    """
    p.L.info("Allocating national overnight stays based on hotel points.")

    # Define input paths relative to data_dir
    overnight_path = os.path.join(p.data_dir, 'gep_recreation/0_inputs/unwto-all-data-download_2022.xlsx')
    country_gpkg_path = os.path.join(p.data_dir, 'gep_recreation/0_inputs/ee_r250_correspondence.gpkg')
    ref_raster_path = os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/grip4_road_length/grip4_road_length_1km.tif')
    hotel_raster_path = os.path.join(p.cur_dir, 'hotels_1km.tif')

    # Define output path
    output_path = os.path.join(p.cur_dir, 'overnights_1km.tif')
    
    if os.path.exists(output_path):
        p.L.info(f"Overnight raster already exists: {output_path}")
        return 
    
    # Workflow
    p.L.info("Rasterizing hotel data...")
    overnight_df = rasterize_hotel_points(p, hotel_raster_path)

    p.L.info("Importing UNTWO overnight data...")
    overnight_df = clean_untwo_data(p, overnight_path)

    p.L.info(f"Rasterizing country borders...")
    country_raster_path = rasterize_country_borders(p, country_gpkg_path, 'iso3_r250_ids_1km.tif', ref_raster_path, 'iso3_r250_id')

    # Create overnight allocation raster using merged data
    create_overnight_allocation_raster(p, overnight_df, hotel_raster_path, country_raster_path, output_path)
    p.L.info(f"Overnight allocation complete: {output_path}")

def rasterize_hotel_points(p, hotel_raster_path):
    p.L.info("Rasterizing hotel points.")
    
    # Define input paths relative to data_dir
    input_path = os.path.join(p.data_dir, 'gep_recreation/0_inputs/Export_GEO-data_4-University-of-Minnesota.gpkg')
    ref_path = os.path.join(p.data_dir, 'gep_recreation/0_processed_raster_inputs/grip4_road_length/grip4_road_length_1km.tif')
    
    # Define output path
    output_path = hotel_raster_path
    if os.path.exists(output_path):
        p.L.info(f"Hotel raster already exists: {output_path}")
        return 
    
    ref_info = pygeoprocessing.geoprocessing.get_raster_info(ref_path)
    pygeoprocessing.geoprocessing.create_raster_from_bounding_box(
        target_bounding_box=ref_info['bounding_box'], 
        target_raster_path=output_path, 
        target_pixel_size=ref_info['pixel_size'], 
        target_pixel_type=gdal.GDT_Int16, 
        target_srs_wkt=ref_info['projection_wkt'], 
        target_nodata=-1, 
        fill_value=0
    )
    pygeoprocessing.geoprocessing.rasterize(
        vector_path=input_path, 
        target_raster_path=output_path, 
        burn_values=[1], 
        option_list=['ALL_TOUCHED=TRUE', 'MERGE_ALG=ADD'], 
    )

def extract_clean_overnights(df, tourism_type):

    df = df.copy()
    df.columns = df.columns.map(str)
    iso3_r250_id_col = "C."
    country_col = "Basic data and indicators"
    indicator_col = "Unnamed: 6"
    type_col = "Unnamed: 5"
    df[iso3_r250_id_col] = df[iso3_r250_id_col].ffill()
    df[country_col] = df[country_col].ffill()
    df[indicator_col] = df[indicator_col].ffill()
    df[type_col] = df[type_col].ffill()

    # Filter only overnights rows
    overnight_df = df[df[indicator_col].str.contains("Overnights", na=False)]

    # Keep only relevant columns
    year_cols = [col for col in df.columns if col.isdigit()]
    overnight_df = overnight_df[[iso3_r250_id_col, country_col, type_col] + year_cols]

    # Melt to long format
    tidy = overnight_df.melt(
        id_vars=[iso3_r250_id_col, country_col, type_col],
        var_name="year",
        value_name="overnights"
    )

    tidy.rename(columns={iso3_r250_id_col: "iso3_r250_id", country_col: "unwto_name", type_col: "overnight_type"}, inplace=True)
    tidy["tourism_type"] = tourism_type
    tidy["year"] = tidy["year"].astype(int)
    tidy["overnights"] = pd.to_numeric(tidy["overnights"], errors="coerce")

    return tidy

def clean_untwo_data(p, overnight_path):
    # Find starting row (where 'AFGHANISTAN' appears)
    domestic_start = pd.read_excel(overnight_path, sheet_name='Domestic Tourism-Accommodation')
    inbound_start = pd.read_excel(overnight_path, sheet_name='Inbound Tourism-Accommodation')
    domestic_start_idx = domestic_start[domestic_start.iloc[:, 3] == 'AFGHANISTAN'].index[0]
    inbound_start_idx = inbound_start[inbound_start.iloc[:, 3] == 'AFGHANISTAN'].index[0]
    domestic_data = pd.read_excel(overnight_path, sheet_name='Domestic Tourism-Accommodation', skiprows=domestic_start_idx)
    inbound_data = pd.read_excel(overnight_path, sheet_name='Inbound Tourism-Accommodation', skiprows=inbound_start_idx)

    # Run for both & combine
    domestic_df = extract_clean_overnights(domestic_data, "domestic")
    inbound_df = extract_clean_overnights(inbound_data, "international")
    panel_df = pd.concat([domestic_df, inbound_df], ignore_index=True)

    # Normalize types (in case of spacing or case issues)
    panel_df["overnight_type"] = panel_df["overnight_type"].str.lower().str.strip()

    # Pivot
    pivoted = panel_df.pivot_table(
        index=["iso3_r250_id", "unwto_name", "year"],
        columns=["overnight_type", "tourism_type"],
        values="overnights",
        aggfunc="first",
    ).reset_index()

    # Flatten MultiIndex columns
    pivoted.columns = ['iso3_r250_id', 'unwto_name', 'year'] + [
        f"{otype}_overnights_{ttype}" for otype, ttype in pivoted.columns[3:]
    ]

    # Use Total if available, otherwise use Hotels
    for ttype in ['domestic', 'international']:
        total_col = f"total_overnights_{ttype}"
        hotel_col = f"hotels and similar establishments_overnights_{ttype}"
        combined_col = f"overnights_{ttype}"
        pivoted[combined_col] = pivoted[total_col].fillna(0) + pivoted[hotel_col].fillna(0)
        # But prevent double counting:
        pivoted[combined_col] = np.where(
            pivoted[total_col].notna(), 
            pivoted[total_col],  # Use total if available
            pivoted[hotel_col]   # Use hotels if total missing
        )

    # Reorder and rename columns
    final_cols = ['iso3_r250_id', 'unwto_name', 'year']
    for ttype in ['domestic', 'international']:
        final_cols += [
            f"total_overnights_{ttype}",
            f"hotel_overnights_{ttype}",
            f"overnights_{ttype}"
        ]

    pivoted.rename(columns={
        'hotels and similar establishments_overnights_domestic': 'hotel_overnights_domestic',
        'hotels and similar establishments_overnights_international': 'hotel_overnights_international'
    }, inplace=True)

    # Filter to final cols that exist
    final_cols = [col for col in final_cols if col in pivoted.columns]
    final_df = pivoted[final_cols]
    output_path = os.path.join(p.cur_dir, "unwto_panel.csv")
    final_df.to_csv(output_path, index=False)
    return final_df

def rasterize_country_borders(p, vector_path, raster_name, ref_raster_path, column):
    # Define output path
    output_path = os.path.join(p.cur_dir, raster_name)
    if os.path.exists(output_path):
        p.L.info(f"Rasterized borders already exists: {output_path}")
        return output_path
    
    ref_info = pygeoprocessing.geoprocessing.get_raster_info(ref_raster_path)

    pygeoprocessing.geoprocessing.create_raster_from_bounding_box(
        target_bounding_box=ref_info['bounding_box'], 
        target_raster_path=output_path, 
        target_pixel_size=ref_info['pixel_size'], 
        target_pixel_type=ref_info['datatype'], 
        target_srs_wkt=ref_info['projection_wkt'], 
        target_nodata=ref_info['nodata'][0], 
        fill_value=0
    )

    pygeoprocessing.geoprocessing.rasterize(
        vector_path=vector_path, 
        target_raster_path=output_path, 
        burn_values=None, 
        option_list=['ALL_TOUCHED=TRUE', f'ATTRIBUTE={column}'], 
    )

    return output_path

def create_overnight_allocation_raster(p, overnight_df, hotel_raster_path, country_raster_path, output_path, target_year=2019):
    """
    Create overnight allocation raster based on hotel distribution
    Args:
        p: Parameter object with logging
        overnight_df: DataFrame with overnight stay data already merged with iso3_r250_id
        hotel_raster_path: Path to hotel density raster
        country_raster_path: Path to country ID raster
        output_path: Path for output overnight allocation raster
        target_year: Year to use for overnight data (default: 2019)
    """
    p.L.info("Creating country-to-overnight mapping...")

    # Filter to target year
    year_data = overnight_df[overnight_df['year'] == target_year].copy()

    # Create iso3_r250_id to overnights mapping directly from merged data
    country_overnights_map = {}
    total_overnights_allocated = 0
    for _, row in year_data.iterrows():
        iso3_r250_id = row['iso3_r250_id']
        # Use combined overnights (domestic + international)
        total_overnights = 0
        for col in ['overnights_domestic', 'overnights_international']:
            if col in row and pd.notna(row[col]):
                total_overnights += row[col]
        if total_overnights > 0:
            # Sum if multiple entries for same country (shouldn't happen but just in case)
            if iso3_r250_id in country_overnights_map:
                country_overnights_map[iso3_r250_id] += total_overnights
            else:
                country_overnights_map[iso3_r250_id] = total_overnights
            total_overnights_allocated += total_overnights

    mapped_countries = len(country_overnights_map)
    p.L.info(f"Successfully mapped overnight data for {mapped_countries} countries")
    p.L.info(f"Total overnight stays allocated: {total_overnights_allocated:,.0f}")

    # Allocate overnights to hotel pixels
    p.L.info("Allocating overnights to hotel pixels...")

    def allocate_overnights_op(hotel_array, country_array):
        """Allocate country overnights proportionally to hotel density within each country"""
        result = np.zeros_like(hotel_array, dtype=np.float32)
        for ee_id, total_overnights in country_overnights_map.items():
            country_mask = (country_array == ee_id)
            hotels_in_country = np.sum(hotel_array[country_mask])
            if hotels_in_country > 0:
                # Allocate overnights proportional to hotel density
                result[country_mask] = (hotel_array[country_mask] / hotels_in_country) * total_overnights
        return result
    
    # Execute the raster calculation
    pygeoprocessing.raster_calculator(
        [(hotel_raster_path, 1), (country_raster_path, 1)],
        allocate_overnights_op,
        output_path,
        gdal.GDT_Float32,
        -9999
    )
    p.L.info(f"Overnight allocation raster created: {output_path}")



def calculate_tourist_recreation_into_sites_kernel(p):
    """
    Calculate daily recreation visits and values flowing INTO recreation sites
    using kernel-based convolution for better performance on large rasters.
    """
    # Define input paths
    inputs = {
        'hq_sites': os.path.join(p.recreation_site_analysis_dir, 'high_quality_sites_1km.tif'),
        'population': os.path.join(p.allocate_overnights_dir, 'overnights_1km.tif'),
        'country_id': os.path.join(p.allocate_overnights_dir, 'iso3_r250_ids_1km.tif'),
        'country_costs': os.path.join(p.data_dir, 'gep_recreation/0_inputs/feul_cost_per_km_2019_2datasources_iso3.csv')
    }
    
    # Define output paths
    outputs = {
        'visits': os.path.join(p.cur_dir, 'site_visits_1km.tif'),
        'value': os.path.join(p.cur_dir, 'site_value_1km.tif'),
    }
    
    # Check if outputs already exist
    if all(os.path.exists(output) for output in outputs.values()):
        p.L.info("Recreation site visit outputs already exist, skipping calculation")
        return
    
    # Check if all input files exist
    missing_files = []
    for key, path in inputs.items():
        if not os.path.exists(path):
            missing_files.append(f"{key}: {path}")
    
    if missing_files:
        p.L.error(f"Missing input files:\n" + "\n".join(missing_files))
        raise FileNotFoundError("Missing required input files")
    
    p.L.info("Starting kernel-based recreation site calculation")
    
    # Calculate for each distance buffer and sum results
    calculate_kernel_based_recreation(inputs, outputs, p)

def aggregate_results_iso3(p):
    """
    Aggregate daily and tourist recreation results to country levels.
    """
    # Define input paths
    inputs = {
        'daily_visits': os.path.join(p.calculate_daily_recreation_into_sites_kernel_dir, 'site_visits_1km.tif'),
        'daily_value': os.path.join(p.calculate_daily_recreation_into_sites_kernel_dir, 'site_value_1km.tif'),
        'tourist_visits': os.path.join(p.calculate_tourist_recreation_into_sites_kernel_dir, 'site_visits_1km.tif'),
        'tourist_value': os.path.join(p.calculate_tourist_recreation_into_sites_kernel_dir, 'site_value_1km.tif'),
        'country_id': os.path.join(p.allocate_overnights_dir, 'iso3_r250_ids_1km.tif'),
        'country_gpkg': os.path.join(p.data_dir, 'gep_recreation', '0_inputs', 'ee_r250_correspondence.gpkg'),
    }
    
    # Define output paths
    outputs = {
        'results_by_country_csv': os.path.join(p.cur_dir, 'results_by_country.csv'),
        'results_by_country_gpkg': os.path.join(p.cur_dir, 'results_by_country.gpkg'),
    }
    
    # Check if outputs already exist
    if all(os.path.exists(output) for output in outputs.values()):
        p.L.info("Aggregated results already exist, skipping aggregation")
        return
    
    # Load country_id raster as array
    try:
        # Load country_id raster as array
        country_id_array = pygeoprocessing.raster_to_numpy_array(inputs['country_id'])
        ndv = pygeoprocessing.get_raster_info(inputs['country_id'])['nodata'][0]
    except Exception as e:
        p.L.error(f"Failed to load country_id raster: {e}")
    # country_id_array = pygeoprocessing.raster_to_numpy_array(inputs['country_id'])
    # ndv = pygeoprocessing.get_raster_info(inputs['country_id'])['nodata'][0]
    valid_mask = (country_id_array != ndv) & (~np.isnan(country_id_array))
    country_ids = np.unique(country_id_array[valid_mask])
    country_ids = country_ids[country_ids > 0]  # Only positive IDs

    # For each country, sum values from each raster
    rows = []
    total_countries = len(country_ids)
    results_arrays = {}
    for key in ['daily_visits', 'daily_value', 'tourist_visits', 'tourist_value']:
        arr = pygeoprocessing.raster_to_numpy_array(inputs[key])
        arr_ndv = pygeoprocessing.get_raster_info(inputs[key])['nodata'][0]
        arr = np.where(arr == arr_ndv, np.nan, arr)
        results_arrays[key] = arr

    # Loop through countries and aggregate
    for idx, cid in enumerate(country_ids, 1):
        mask = (country_id_array == cid)
        row = {'iso3_r250_id': int(cid)}
        if idx % 10 == 0 or idx == 1 or idx == total_countries:
            p.L.info(f"Processed {idx}/{total_countries} countries")
        for key in results_arrays:
            val = np.nansum(results_arrays[key][mask])
            row[key] = val
        rows.append(row)

    # Merge with country polygons and save as gpkg
    df = pd.DataFrame(rows)
    gdf = gpd.read_file(inputs['country_gpkg'])
    gdf_merged = gdf.merge(df, on='iso3_r250_id', how='left')
    gdf_out = gdf_merged[['iso3_r250_id', 'iso3_r250_label', 'daily_visits', 'daily_value', 'tourist_visits', 'tourist_value', 'geometry']]

    # Save to CSV
    df_out = pd.DataFrame(gdf_out)
    df_out = df_out.drop(columns='geometry')
    df_out.to_csv(outputs['results_by_country_csv'], index=False)
    p.L.info(f"Saved country results to {outputs['results_by_country_csv']}")

    # Merge with country polygons and save as gpkg
    gdf_out.to_file(outputs['results_by_country_gpkg'], driver='GPKG')
    p.L.info(f"Saved country results to {outputs['results_by_country_gpkg']}")
