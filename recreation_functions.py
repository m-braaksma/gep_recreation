"""
recreation_functions.py

Scientific functions for the recreation ecosystem service: site
classification, overnight allocation, the distance-banded demand model, and
valuation. Functions take explicit paths and parameters (no ProjectFlow
object); recreation_tasks.py wires them into the task tree.
"""

import os
import shutil
import numpy as np
import pandas as pd
import pygeoprocessing
from osgeo import gdal, osr
from scipy import ndimage
import hazelbean as hb

import recreation_utils as ru

L = hb.get_logger('recreation_functions')


# ===========================================================================
# Model constants
# ===========================================================================

TARGET_YEAR = 2019

# Nearest-year fallbacks to TARGET_YEAR farther than this are treated as
# missing rather than silently used (some UNWTO series otherwise fall back
# as far as 1995).
MAX_YEAR_GAP = 5

# Nature-attribution share (lambda) for accommodation and air travel: the
# EU27-average share of personal travellers citing "natural environment" as
# the single most important reason for choosing their destination (Flash
# Eurobarometer 499, 2021). Applied globally as a single anchor; the survey's
# top-two-reasons share (0.426) is the reported upper bound. Ground-travel
# value keeps lambda=1 since those flows are already restricted to
# high-quality natural sites by construction.
NATURE_ATTRIBUTION_SHARE = 0.183

# Day-trip demand model
MAX_DISTANCE_KM = 50.0
MAX_BANDS = 8
KM_PER_DEGREE = 111.32

# Aligned WorldPop rasters were resampled from 30 arc-second pixels to 0.01
# degree without rescaling the per-pixel counts, so each pixel holds the
# count of a ~1.44x smaller area (raster sum 5.54B vs ~7.8B world population
# in 2020). Rescale by the pixel-area ratio.
WORLDPOP_AREA_SCALE = (0.01 / (30.0 / 3600.0)) ** 2


# ===========================================================================
# Recreation sites
# ===========================================================================

def environment_index(input_paths, output_path):
    """input_paths: dict with cropland, forest, grassland, othernat, urban,
    water and pa share rasters."""
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

    base_list = [(input_paths[k], 1) for k in ['cropland', 'forest', 'grassland', 'othernat', 'urban', 'water', 'pa']]
    pygeoprocessing.raster_calculator(base_list, environment_index_op, output_path,
                                       gdal.GDT_Int32, nodata_value, calc_raster_stats=True)


def accessibility_index(urban_path, road_path, output_path):
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

    pygeoprocessing.raster_calculator([(urban_path, 1), (road_path, 1)], accessibility_op,
                                       output_path, gdal.GDT_Int32, nodata_value)


def rank_sites(accessibility_path, env_index_path, output_path):
    site_matrix = np.array([
        [1, 1, 4, 7], [1, 4, 4, 7], [2, 2, 8, 8], [3, 5, 5, 9], [3, 6, 6, 9]])
    nodata_value = -1

    def site_op(acc_class, env_class):
        acc_class = acc_class - 1
        return site_matrix[acc_class, env_class].astype(np.int16)

    pygeoprocessing.raster_calculator([(accessibility_path, 1), (env_index_path, 1)], site_op,
                                       output_path, gdal.GDT_Int32, nodata_value)


def high_quality_sites(ranked_sites_path, output_path):
    """Sites ranked 9: top environment class with medium-high/high access."""
    def hq_op(site_class):
        return np.where(site_class == 9, 1, 0).astype(np.int32)

    pygeoprocessing.raster_calculator([(ranked_sites_path, 1)], hq_op, output_path, gdal.GDT_Int32, -1)


def dissolve_sites(hq_sites_path, output_path, site_sizes_path):
    """
    Dissolve contiguous high-quality-site pixels into discrete sites
    (8-connected), so a 65-pixel park is 1 destination, not 65.

    Output raster: 0 = not a site, 1..N = site id. Also writes per-site
    pixel counts as a fragmentation check.
    """
    hq_nodata = pygeoprocessing.get_raster_info(hq_sites_path)['nodata'][0]
    hq_arr = ru.read_raster_array(hq_sites_path)

    is_hq = (hq_arr == 1) if hq_nodata is None else ((hq_arr != hq_nodata) & (hq_arr == 1))
    structure = np.ones((3, 3))  # 8-connectivity; use ndimage default (4-conn) if diagonal touches shouldn't merge sites
    labeled, n_sites = ndimage.label(is_hq, structure=structure)
    L.info(f"Found {n_sites} discrete high-quality sites "
           f"({int(is_hq.sum())} HQ pixels, {is_hq.sum()/max(n_sites,1):.1f} px/site avg)")

    pygeoprocessing.new_raster_from_base(hq_sites_path, output_path, gdal.GDT_Int32, [0])
    out_ds = gdal.Open(output_path, gdal.GA_Update)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(labeled.astype(np.int32))
    out_band.FlushCache()
    out_band, out_ds = None, None

    site_ids, pixel_counts = np.unique(labeled[labeled > 0], return_counts=True)
    pd.DataFrame({'site_id': site_ids, 'pixel_count': pixel_counts}).to_csv(site_sizes_path, index=False)


# ===========================================================================
# Overnight allocation
# ===========================================================================

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
    # Every Overnights row in the workbook has Units = "Thousands".
    tidy["overnights"] = pd.to_numeric(tidy["overnights"], errors="coerce") * 1000.0
    return tidy


def clean_unwto_overnights(workbook_path):
    """Domestic + inbound overnights panel (country x year), preferring
    total overnights over hotel-only overnights where both exist."""
    domestic_df = extract_clean_overnights(ru.read_unwto_sheet(workbook_path, 'Domestic Tourism-Accommodation'), "domestic")
    inbound_df = extract_clean_overnights(ru.read_unwto_sheet(workbook_path, 'Inbound Tourism-Accommodation'), "international")
    panel_df = pd.concat([domestic_df, inbound_df], ignore_index=True)
    panel_df = ru.recode_unwto_ids(panel_df)
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
    return pivoted[final_cols]


def country_overnights(overnight_df, target_year=TARGET_YEAR):
    """
    Country -> total overnights, using the nearest year to target_year with
    data (UNWTO reporting is patchy), within MAX_YEAR_GAP.

    Returns (country_overnights_map, substitution_report DataFrame).
    """
    df = overnight_df.copy()
    df['total_overnights'] = df[['overnights_domestic', 'overnights_international']].sum(axis=1, min_count=1)
    df = df[df['total_overnights'] > 0]
    # several UNWTO entities can share one r250 id after recoding (e.g. Saba
    # and Sint Eustatius -> 535), so combine them per country-year
    df = df.groupby(['iso3_r250_id', 'year'], as_index=False).agg(
        unwto_name=('unwto_name', lambda s: ' + '.join(sorted(set(s)))),
        total_overnights=('total_overnights', 'sum'))

    report = ru.pick_nearest_year(df, ['unwto_name', 'total_overnights'], target_year)
    report = report.rename(columns={'year': 'year_used'})
    report['target_year'] = target_year
    report['used'] = report['year_distance'] <= MAX_YEAR_GAP
    used = report[report['used']]
    country_overnights_map = dict(zip(used['iso3_r250_id'], used['total_overnights']))

    exact_year_count = int((used['year_distance'] == 0).sum())
    too_old = sorted(report.loc[~report['used'], 'iso3_r250_id'].tolist())
    L.info(f"Mapped overnight data for {len(country_overnights_map)} countries "
           f"({exact_year_count} with exact {target_year} data, "
           f"{len(country_overnights_map) - exact_year_count} using a substitute year); "
           f"{len(too_old)} countries excluded because their nearest year is more than "
           f"{MAX_YEAR_GAP} years from {target_year}: {too_old}")
    L.info(f"Total overnight stays allocated: {sum(country_overnights_map.values()):,.0f}")
    return country_overnights_map, report.sort_values('year_distance', ascending=False)


def allocate_overnights_to_hotels(country_overnights_map, hotel_raster_path, country_raster_path, output_path):
    """Spread each country's overnights across its pixels in proportion to
    hotel counts."""
    # bugfix: compute the total hotel weight per country across the whole raster
    # before calling raster_calculator, not inside the callback.
    # raster_calculator invokes its callback once per internal memory block
    # (not once on the whole raster), so computing this sum from the
    # block-local hotel_array inside the callback silently
    # undercounts it for any country spanning more than one block - which
    # then causes that country's entire national total_overnights to be
    # redundantly re-allocated in full to every block it touches, inflating
    # large countries by roughly however many blocks they span (confirmed:
    # US/Canada/China came out 80-180x too high, small single-block
    # countries were correct).
    hotel_full = ru.read_raster_array(hotel_raster_path).astype(np.float64)
    country_full = ru.read_raster_array(country_raster_path)
    hotels_in_country_map = {
        ee_id: float(hotel_full[country_full == ee_id].sum())
        for ee_id in country_overnights_map
    }
    del hotel_full, country_full

    def allocate_overnights_op(hotel_array, country_array):
        result = np.zeros_like(hotel_array, dtype=np.float32)
        for ee_id, total_overnights in country_overnights_map.items():
            hotels_in_country = hotels_in_country_map[ee_id]
            if hotels_in_country > 0:
                country_mask = (country_array == ee_id)
                result[country_mask] = (hotel_array[country_mask] / hotels_in_country) * total_overnights
        return result

    pygeoprocessing.raster_calculator([(hotel_raster_path, 1), (country_raster_path, 1)],
                                       allocate_overnights_op, output_path, gdal.GDT_Float32, -9999)


# ===========================================================================
# Demand parameters
# ===========================================================================

def load_recreation_params(param_table_path):
    """
    Load and validate the external parameter table.

    Required columns: iso3_r250_id, iso3_r250_name, param_group,
                       participation_param, distance_param

    Returns:
      country_to_group_map : dict, iso3_r250_id -> param_group
      group_params          : dict, param_group -> {'a': participation_param,
                                                      'b': distance_param}
    Raises ValueError if any param_group has inconsistent parameter values
    across the countries assigned to it.
    """
    df = pd.read_csv(param_table_path)
    required = ['iso3_r250_id', 'iso3_r250_name', 'param_group', 'participation_param', 'distance_param']
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Parameter table missing required columns: {missing_cols}")

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

    L.info(f"Loaded recreation demand parameters: {len(group_params)} param_group(s), "
           f"{len(country_to_group_map)} countries")
    return country_to_group_map, group_params


def build_group_id_raster(country_id_raster_path, country_to_group_map, output_path):
    """Remap a country-id raster into a param_group id raster."""
    country_nodata = pygeoprocessing.get_raster_info(country_id_raster_path)['nodata'][0]
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


# ===========================================================================
# Distance-banded demand model
# ===========================================================================

def compute_group_bands(pixel_size_km, max_distance_km=MAX_DISTANCE_KM, max_bands=MAX_BANDS):
    """
    Cutoff banding, grid-aware:
      cutoff_km  = max_distance_km            -- a manually-set farthest
                   reasonable day-trip distance, the SAME for every param
                   group regardless of that group's own decay rate b.
      n_bands    = floor(cutoff_km / pixel_size_km), clamped to [1, max_bands]
                   -- guarantees every band is at least one pixel wide.
      band_width = cutoff_km / n_bands        -- NOT snapped to the discrete
                   set of achievable pixel-offset radii; band membership is
                   therefore approximate at each band's edge.

    b is still used (cdf_exp) to shape how much of the fixed cutoff's demand
    falls in each band; it just doesn't set the cutoff itself.

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


def build_ring_kernel(r_lo, r_hi, dx_km, dy_km, kernel_dir, tag):
    """
    Binary indicator kernel: 1 for pixel offsets whose distance falls in
    (r_lo, r_hi] (or [0, r_hi] for the first band, r_lo=0), else 0. Used
    for BOTH detecting site presence within the band (denom) and spreading
    a band's fixed visit/value pool evenly across whatever sites are present.

    dx_km/dy_km are the east-west/north-south pixel sizes; on a geographic
    grid dx_km shrinks with cos(latitude), so the ring is an ellipse in
    pixel space.
    """
    rx = int(np.ceil(r_hi / dx_km))
    ry = int(np.ceil(r_hi / dy_km))
    yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
    d_km = np.sqrt((xx * dx_km) ** 2 + (yy * dy_km) ** 2)
    within = (d_km <= r_hi) if r_lo == 0 else ((d_km > r_lo) & (d_km <= r_hi))
    kernel = within.astype(np.float32)
    kernel_path = os.path.join(kernel_dir, f'ring_kernel_{tag}.tif')
    ru.write_array_as_raster(kernel, dy_km, kernel_path)
    return kernel_path


def load_fuel_cost_map(country_costs_path):
    """iso3_r250_id -> 2019 gasoline cost (USD per km)."""
    cost_df = pd.read_csv(country_costs_path)
    # unit bugfix: the source CSV's gasoline_cost_usd_per_km columns are 100x
    # too high. Traced to clean_cost_per_distance_01_feul_price.R (same
    # 0_inputs folder), which multiplies price-per-liter by a constant
    # `global_liter_consumption_per_km <- 7.1` -- but 7.1 is liters per
    # *100 km* (a standard average-car fuel-economy figure, per the script's
    # own cited IEA source), not liters per km, and the formula never divides
    # by 100. Confirmed: (cost_per_km / price_per_liter) == 7.1 for every one
    # of the 169 countries in the file, with zero variation.
    # Remove this /100 once the shared source CSV is regenerated correctly.
    cost_df['gasoline_cost_usd_per_km_2019_gppdata'] = cost_df['gasoline_cost_usd_per_km_2019_gppdata'] / 100.0
    return dict(zip(cost_df['iso3_r250_id'], cost_df['gasoline_cost_usd_per_km_2019_gppdata']))


def calculate_recreation_flows(inputs, outputs, group_params, work_dir,
                               max_distance_km=MAX_DISTANCE_KM, max_bands=MAX_BANDS):
    """
    Core flow model, banded version. For every param_group g with (a, b):
      cutoff_km_g = max_distance_km     (fixed farthest reasonable day-trip
                     distance, the same for every group)
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
        unconditionally

    MEMORY: never loads a full global raster into memory. Only group_id gets
    one streaming pass (scan_group_bboxes); everything else is read as
    windowed tiles directly off disk.

    inputs: dict with 'hq_sites', 'hq_sites_dissolved', 'population',
            'group_id', 'country_id', 'country_costs' (csv path), and
            optional 'population_scale'
    outputs: dict with 'visits', 'value', 'unmet' (raster paths),
             'site_table', 'group_table', 'missing_cost_iso3' (csv paths)
    """
    kernel_dir = os.path.join(work_dir, 'kernels')
    os.makedirs(kernel_dir, exist_ok=True)

    L.info("Verifying all input rasters share the population raster's pixel grid")
    ru.assert_rasters_aligned(inputs['population'], {
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
    srs = osr.SpatialReference()
    srs.ImportFromWkt(raster_info['projection_wkt'])
    is_geographic = bool(srs.IsGeographic())
    geotransform = raster_info['geotransform']
    if is_geographic:
        dy_km = px_y * KM_PER_DEGREE
    else:
        dy_km = ru.convert_pixel_size_to_km(inputs['population'], pixel_size_native)

    def tile_dx_km(row_lo, row_hi):
        # east-west pixel width at the tile's center latitude; floored so
        # near-polar rows don't produce enormous kernels
        if not is_geographic:
            return dy_km
        center_lat = geotransform[3] + 0.5 * (row_lo + row_hi + 1) * geotransform[5]
        return max(px_x * KM_PER_DEGREE * np.cos(np.radians(center_lat)), 0.05 * dy_km)

    population_scale = inputs.get('population_scale', 1.0)
    pop_nodata = raster_info['nodata'][0]
    n_cols, n_rows = raster_info['raster_size']

    cost_map = load_fuel_cost_map(inputs['country_costs'])
    max_cid = max(cost_map.keys()) if cost_map else 0
    cost_lookup = np.zeros(max_cid + 1, dtype=np.float64)
    for cid, cost in cost_map.items():
        if 0 <= cid <= max_cid:
            cost_lookup[cid] = cost

    hq_nodata = pygeoprocessing.get_raster_info(inputs['hq_sites'])['nodata'][0]

    # output rasters: filled with 0 (not nodata) so that windowed
    # read-modify-write ADD works across overlapping buffers
    for key in ['visits', 'value', 'unmet']:
        pygeoprocessing.new_raster_from_base(
            inputs['population'], outputs[key], gdal.GDT_Float32, [-9999])
        ds = gdal.Open(outputs[key], gdal.GA_Update)
        ds.GetRasterBand(1).Fill(0)
        ds = None

    L.info("Scanning parameter-group extents")
    bboxes = ru.scan_group_bboxes(inputs['group_id'])
    L.info(f"Processing {len(bboxes)} parameter group(s)")

    ru.log_country_coverage(inputs['country_id'], group_params.keys(), 'param_group',
                            'contributing exactly zero visits/value/unmet',
                            group_id_path=inputs['group_id'])
    _, uncovered_cost_iso3 = ru.log_country_coverage(
        inputs['country_id'], cost_map.keys(), 'fuel-cost',
        'value computed as if travel is free there -- flagged for NaN downstream')
    pd.DataFrame({'iso3_r250_id': sorted(uncovered_cost_iso3)}).to_csv(
        outputs['missing_cost_iso3'], index=False)

    group_qa_rows = []

    # persistent, reused across groups -- windowed reads only
    pop_ds = gdal.Open(inputs['population']); pop_band = pop_ds.GetRasterBand(1)
    country_ds = gdal.Open(inputs['country_id']); country_band = country_ds.GetRasterBand(1)
    hq_ds = gdal.Open(inputs['hq_sites']); hq_band = hq_ds.GetRasterBand(1)
    group_ds = gdal.Open(inputs['group_id']); group_band = group_ds.GetRasterBand(1)

    visits_out_ds = gdal.Open(outputs['visits'], gdal.GA_Update); visits_out_band = visits_out_ds.GetRasterBand(1)
    value_out_ds = gdal.Open(outputs['value'], gdal.GA_Update); value_out_band = value_out_ds.GetRasterBand(1)
    unmet_out_ds = gdal.Open(outputs['unmet'], gdal.GA_Update); unmet_out_band = unmet_out_ds.GetRasterBand(1)

    # A group's bbox can span nearly the whole raster (e.g. one global param
    # group), and a single windowed read that large is tens of GB. Split each
    # group's bbox into core tiles, each read with its own cutoff buffer; a
    # core-tile mask ensures each population pixel's demand is generated by
    # exactly one tile even though buffered windows overlap.
    # Tiles are short in latitude (TILE_ROWS) so one cos(latitude)-scaled
    # east-west pixel width per tile is accurate to within a few percent
    # (500 rows = 5 degrees at 0.01 deg pixels).
    TILE_ROWS = 500 if is_geographic else 5000
    TILE_COLS = 12000

    for gid, (r0, r1, c0, c1) in bboxes.items():
        params = group_params.get(gid)
        if params is None:
            L.warning(f"No parameters for param_group {gid}, skipping")
            continue
        a, b = params['a'], params['b']

        cutoff_km_g, edges = compute_group_bands(dy_km, max_distance_km, max_bands)
        n_bands = len(edges) - 1
        beyond_cutoff_share = 1.0 - cdf_exp(cutoff_km_g, b)

        group_budget_total = 0.0
        group_realized_total = 0.0
        group_unmet_total = 0.0

        tiles = list(ru.iter_core_tiles(r0, r1, c0, c1, TILE_ROWS, TILE_COLS))
        L.info(f"Param group {gid}: processing {len(tiles)} tiles of up to "
               f"{TILE_ROWS}x{TILE_COLS} pixels (tiles with no population are skipped)")
        kernel_cache = {}

        for tile_index, (tr0, tr1, tc0, tc1) in enumerate(tiles):
            dx_km = tile_dx_km(tr0, tr1)
            buffer_px_y = int(np.ceil(cutoff_km_g / dy_km))
            buffer_px_x = int(np.ceil(cutoff_km_g / dx_km))
            r0b, r1b = max(tr0 - buffer_px_y, 0), min(tr1 + buffer_px_y + 1, n_rows)
            c0b, c1b = max(tc0 - buffer_px_x, 0), min(tc1 + buffer_px_x + 1, n_cols)
            xoff, yoff = c0b, r0b
            wx, wy = c1b - c0b, r1b - r0b

            group_sub = ru.read_window(group_band, xoff, yoff, wx, wy)
            pop_sub = ru.read_window(pop_band, xoff, yoff, wx, wy).astype(np.float64)

            # core_mask: True only for pixels this tile owns; the buffer
            # margin is read so ring convolutions can see sites just across
            # the tile edge, but must not itself generate demand here.
            row_idx = np.arange(yoff, yoff + wy)[:, None]
            col_idx = np.arange(xoff, xoff + wx)[None, :]
            core_mask = (row_idx >= tr0) & (row_idx <= tr1) & (col_idx >= tc0) & (col_idx <= tc1)

            group_mask = (group_sub == gid) & core_mask
            valid_sub = (pop_sub != pop_nodata) & ~np.isnan(pop_sub) & (pop_sub > 0) & group_mask
            if not valid_sub.any():
                continue

            country_sub = ru.read_window(country_band, xoff, yoff, wx, wy)
            hq_sub = ru.read_window(hq_band, xoff, yoff, wx, wy)

            budget = np.where(valid_sub, pop_sub * population_scale * a, 0.0)
            cost_per_km_sub = np.zeros_like(pop_sub, dtype=np.float64)
            in_range = valid_sub & (country_sub >= 0) & (country_sub <= max_cid)
            cost_per_km_sub[in_range] = cost_lookup[country_sub[in_range].astype(int)]

            site_ind_sub = (hq_sub == 1).astype(np.float32) if hq_nodata is None \
                else ((hq_sub != hq_nodata) & (hq_sub == 1)).astype(np.float32)

            tmp_dir = os.path.join(kernel_dir, f'tmp_{gid}_{tile_index}')
            os.makedirs(tmp_dir, exist_ok=True)
            site_ind_path = os.path.join(tmp_dir, 'site_ind.tif')
            ru.write_array_as_raster(site_ind_sub, dy_km, site_ind_path)

            realized_total = np.zeros_like(pop_sub, dtype=np.float64)
            unmet_total = np.zeros_like(pop_sub, dtype=np.float64)
            visits_accum_g = np.zeros_like(pop_sub, dtype=np.float64)
            value_accum_g = np.zeros_like(pop_sub, dtype=np.float64)

            for k in range(n_bands):
                r_lo, r_hi = edges[k], edges[k + 1]
                r_mid = 0.5 * (r_lo + r_hi)
                band_share = cdf_exp(r_hi, b) - cdf_exp(r_lo, b)

                Q_band = np.where(valid_sub, budget * band_share, 0.0)

                kernel_key = (round(dx_km, 6), k)
                if kernel_key not in kernel_cache:
                    kernel_cache[kernel_key] = build_ring_kernel(
                        r_lo, r_hi, dx_km, dy_km, kernel_dir, tag=f'{gid}_{k}_{len(kernel_cache)}')
                ring_kernel_path = kernel_cache[kernel_key]
                denom_path = os.path.join(tmp_dir, f'denom_{k}.tif')
                q_visits_path = os.path.join(tmp_dir, f'q_visits_{k}.tif')
                q_value_path = os.path.join(tmp_dir, f'q_value_{k}.tif')
                visits_conv_path = os.path.join(tmp_dir, f'visits_at_sites_{k}.tif')
                value_conv_path = os.path.join(tmp_dir, f'value_at_sites_{k}.tif')

                # denom_k_i = count of sites reachable within this ring
                pygeoprocessing.convolve_2d(
                    (site_ind_path, 1), (ring_kernel_path, 1), denom_path,
                    ignore_nodata_and_edges=False, mask_nodata=False,
                    target_datatype=gdal.GDT_Float32, target_nodata=0)
                denom_arr = ru.read_raster_array(denom_path)
                denom_safe = np.where(denom_arr > 1e-12, denom_arr, np.nan)

                # split this ring's fixed budget evenly across sites present;
                # 0 sites in ring -> this ring's whole share is unmet
                Q_visits_norm = np.where(np.isfinite(denom_safe), Q_band / denom_safe, 0.0).astype(np.float32)
                Q_value_norm = np.where(np.isfinite(denom_safe),
                                         (Q_band / denom_safe) * cost_per_km_sub * 2.0 * r_mid, 0.0).astype(np.float32)

                no_site_in_ring = ~np.isfinite(denom_safe) & valid_sub
                unmet_total[no_site_in_ring] += Q_band[no_site_in_ring]
                realized_total += np.where(no_site_in_ring, 0.0, Q_band)

                ru.write_array_as_raster(Q_visits_norm, dy_km, q_visits_path)
                ru.write_array_as_raster(Q_value_norm, dy_km, q_value_path)

                pygeoprocessing.convolve_2d(
                    (q_visits_path, 1), (ring_kernel_path, 1), visits_conv_path,
                    ignore_nodata_and_edges=False, mask_nodata=False,
                    target_datatype=gdal.GDT_Float32, target_nodata=0)
                pygeoprocessing.convolve_2d(
                    (q_value_path, 1), (ring_kernel_path, 1), value_conv_path,
                    ignore_nodata_and_edges=False, mask_nodata=False,
                    target_datatype=gdal.GDT_Float32, target_nodata=0)

                visits_accum_g += ru.read_raster_array(visits_conv_path) * site_ind_sub
                value_accum_g += ru.read_raster_array(value_conv_path) * site_ind_sub

            # beyond this group's cutoff: unconditionally unmet
            beyond_cutoff_unmet = np.where(valid_sub, budget * beyond_cutoff_share, 0.0)
            unmet_total += beyond_cutoff_unmet

            group_budget_total += float(budget.sum())
            group_realized_total += float(realized_total.sum())
            group_unmet_total += float(unmet_total.sum())

            ru.accumulate_window(visits_out_band, xoff, yoff, visits_accum_g.astype(np.float32))
            ru.accumulate_window(value_out_band, xoff, yoff, value_accum_g.astype(np.float32))
            ru.accumulate_window(unmet_out_band, xoff, yoff, unmet_total.astype(np.float32))

            # free this tile's temp files now rather than at the very end
            shutil.rmtree(tmp_dir, ignore_errors=True)

            L.info(f"  Param group {gid}: tile {tile_index + 1}/{len(tiles)} done")

        group_qa_rows.append({
            'param_group': gid, 'participation_param_a': a, 'distance_param_b': b,
            'cutoff_km': cutoff_km_g, 'n_bands': n_bands, 'n_tiles': len(tiles),
            'budget_total': group_budget_total,
            'realized_total': group_realized_total,
            'unmet_total': group_unmet_total,
        })

    for band in [visits_out_band, value_out_band, unmet_out_band]:
        band.FlushCache()
    pop_band = None; pop_ds = None
    country_band = None; country_ds = None
    hq_band = None; hq_ds = None
    group_band = None; group_ds = None
    visits_out_band = None; visits_out_ds = None
    value_out_band = None; value_out_ds = None
    unmet_out_band = None; unmet_out_ds = None

    pd.DataFrame(group_qa_rows).to_csv(outputs['group_table'], index=False)
    L.info(f"Saved param_group QA table (budget/realized/unmet) to {outputs['group_table']}")

    write_site_table(inputs['hq_sites'], inputs['country_id'], outputs['visits'], outputs['value'],
                     outputs['site_table'], site_id_path=inputs.get('hq_sites_dissolved'))
    L.info(f"Saved site-level visits/value table to {outputs['site_table']}")


def majority_country(country_ids):
    # ignore nodata/0 so a coastal site isn't assigned to "no country"
    valid = country_ids[country_ids > 0]
    return valid.value_counts().idxmax() if len(valid) else np.nan


def write_site_table(hq_sites_path, country_id_path, visits_path, value_path, output_path, site_id_path=None):
    """
    One row per dissolved site (if hq_sites_dissolved given) or per HQ
    pixel (fallback). Streams over blocks rather than loading full rasters.
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

    has_site_id = site_id_band is not None
    hq_band, country_band, visits_band, value_band, site_id_band = None, None, None, None, None
    hq_ds, country_ds, visits_ds, value_ds, site_id_ds = None, None, None, None, None

    per_pixel = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=['site_id', 'iso3_r250_id', 'visits', 'value'])

    if has_site_id:
        agg = per_pixel.groupby('site_id').agg(
            visits=('visits', 'sum'), value=('value', 'sum'), pixel_count=('visits', 'size'),
            iso3_r250_id=('iso3_r250_id', majority_country)).reset_index()
        agg.to_csv(output_path, index=False)
    else:
        per_pixel.to_csv(output_path, index=False)


# ===========================================================================
# Accommodation value
# ===========================================================================

def accommodation_value_raster(overnights_path, price_1km_path, value_path):
    """
    accommodation_value = overnights * mean hotel price per person per night.
    NaN price cells mean "no price estimate", not "free", and propagate as
    NaN rather than 0.
    """
    overnights_nodata = pygeoprocessing.get_raster_info(overnights_path)['nodata'][0]

    def value_op(overnights, price):
        has_overnights = (overnights != overnights_nodata) & (overnights > 0) if overnights_nodata is not None \
            else (overnights > 0)
        price_missing = np.isnan(price)
        result = np.zeros_like(overnights, dtype=np.float32)
        result[has_overnights & price_missing] = np.nan
        priced = has_overnights & ~price_missing
        result[priced] = overnights[priced] * price[priced]
        return result

    pygeoprocessing.raster_calculator(
        [(overnights_path, 1), (price_1km_path, 1)], value_op, value_path, gdal.GDT_Float32, -9999)


def accommodation_value_by_country(overnights_path, country_id_path, value_path):
    """Country totals of overnights, priced overnights and accommodation value."""
    overnights_nodata = pygeoprocessing.get_raster_info(overnights_path)['nodata'][0]
    country_nodata = pygeoprocessing.get_raster_info(country_id_path)['nodata'][0]
    overnights_ds = gdal.Open(overnights_path); overnights_band = overnights_ds.GetRasterBand(1)
    country_ds = gdal.Open(country_id_path); country_band = country_ds.GetRasterBand(1)
    value_ds = gdal.Open(value_path); value_band = value_ds.GetRasterBand(1)

    totals = {}
    for block_info in pygeoprocessing.iterblocks((overnights_path, 1), offset_only=True):
        xoff, yoff = block_info['xoff'], block_info['yoff']
        wx, wy = block_info['win_xsize'], block_info['win_ysize']
        overnights_block = overnights_band.ReadAsArray(xoff, yoff, wx, wy)
        country_block = country_band.ReadAsArray(xoff, yoff, wx, wy)
        value_block = value_band.ReadAsArray(xoff, yoff, wx, wy)

        has_overnights = (overnights_block != overnights_nodata) & (overnights_block > 0) if overnights_nodata is not None \
            else (overnights_block > 0)
        valid_country = (country_block != country_nodata) & (country_block > 0) if country_nodata is not None \
            else (country_block > 0)
        mask = has_overnights & valid_country
        if not np.any(mask):
            continue

        priced_mask = mask & ~np.isnan(value_block)
        for cid in np.unique(country_block[mask]):
            cid = int(cid)
            cmask = mask & (country_block == cid)
            cpriced = priced_mask & (country_block == cid)
            entry = totals.setdefault(cid, {'total_overnights': 0.0, 'priced_overnights': 0.0, 'accommodation_value': 0.0})
            entry['total_overnights'] += float(overnights_block[cmask].sum())
            entry['priced_overnights'] += float(overnights_block[cpriced].sum())
            entry['accommodation_value'] += float(np.nansum(value_block[cmask]))
    overnights_ds, country_ds, value_ds = None, None, None
    overnights_band, country_band, value_band = None, None, None

    rows = []
    for cid, entry in totals.items():
        coverage_share = entry['priced_overnights'] / entry['total_overnights'] if entry['total_overnights'] > 0 else np.nan
        rows.append({
            'iso3_r250_id': cid, 'total_overnights': entry['total_overnights'],
            'priced_overnights': entry['priced_overnights'], 'accommodation_value': entry['accommodation_value'],
            'price_coverage_share': coverage_share,
        })
    return pd.DataFrame(rows).sort_values('price_coverage_share')


# ===========================================================================
# Arrivals, air share, purpose share and air travel value
# ===========================================================================

def clean_unwto_arrivals_by_region(workbook_path):
    """
    Arrivals by destination country, year and origin region (Africa,
    Americas, East Asia and the Pacific, Europe, Middle East, South Asia,
    Other) from UNWTO's 'Inbound Tourism-Regions' sheet. Source values are in
    thousands.
    """
    df = ru.read_unwto_sheet(workbook_path, 'Inbound Tourism-Regions')
    region_col = 'Unnamed: 6'
    valid_regions = ['Africa', 'Americas', 'East Asia and the Pacific', 'Europe',
                      'Middle East', 'South Asia', 'Other not classified']
    df = df[df[region_col].isin(valid_regions)].copy()
    df['C.'] = df['C.'].ffill()

    year_cols = [c for c in df.columns if isinstance(c, int)]
    tidy = df.melt(id_vars=['C.', region_col], value_vars=year_cols,
                    var_name='year', value_name='arrivals_thousands')
    tidy = tidy.rename(columns={'C.': 'iso3_r250_id', region_col: 'origin_region'})
    tidy['origin_region'] = tidy['origin_region'].replace({'Other not classified': 'Other'})
    tidy = ru.recode_unwto_ids(tidy)
    tidy['arrivals'] = pd.to_numeric(tidy['arrivals_thousands'], errors='coerce') * 1000.0
    tidy = tidy.dropna(subset=['arrivals'])
    return tidy.groupby(['iso3_r250_id', 'origin_region', 'year'], as_index=False)['arrivals'].sum()


def clean_unwto_air_share(workbook_path, crosswalk_path, target_year=TARGET_YEAR):
    """
    Share of each destination country's inbound arrivals that come by air,
    from UNWTO's 'Inbound Tourism-Transport' sheet (Air / Total), nearest
    year to target_year within MAX_YEAR_GAP. Countries with no usable
    transport data get their UNWTO region's median (or the global median),
    flagged as imputed.

    The share is computed within the same country's reporting series
    (TF overnight tourists or VF all visitors), so same-day overland
    visitors in a VF series lower the air share accordingly.
    """
    df = ru.read_unwto_sheet(workbook_path, 'Inbound Tourism-Transport')
    df['C.'] = df['C.'].ffill()
    mode = df['Unnamed: 6'].where(df['Unnamed: 6'].isin(['Air', 'Water', 'Land']))
    mode = mode.fillna(df['Unnamed: 5'].where(df['Unnamed: 5'] == 'Total'))
    df = df.assign(mode=mode)
    df = df[df['mode'].notna()]

    year_cols = [c for c in df.columns if isinstance(c, int)]
    tidy = df.melt(id_vars=['C.', 'mode'], value_vars=year_cols, var_name='year', value_name='arrivals')
    tidy = tidy.rename(columns={'C.': 'iso3_r250_id'})
    tidy['arrivals'] = pd.to_numeric(tidy['arrivals'], errors='coerce')
    tidy = ru.recode_unwto_ids(tidy.dropna(subset=['arrivals']))
    wide = tidy.pivot_table(index=['iso3_r250_id', 'year'], columns='mode',
                            values='arrivals', aggfunc='sum').reset_index()
    wide = wide[wide['Air'].notna() & (wide['Total'] > 0)]
    wide['air_share'] = (wide['Air'] / wide['Total']).clip(0.0, 1.0)

    best = ru.pick_nearest_year(wide, ['air_share'], target_year)
    measured = best[best['year_distance'] <= MAX_YEAR_GAP].rename(columns={'year': 'air_share_year'})
    out = ru.fill_by_region(measured[['iso3_r250_id', 'air_share', 'air_share_year']],
                            'air_share', crosswalk_path, 'air_share_imputed')
    return out[['iso3_r250_id', 'air_share', 'air_share_year', 'air_share_imputed']]


def clean_unwto_personal_share(workbook_path, crosswalk_path, target_year=TARGET_YEAR):
    """
    Share of each destination country's inbound arrivals travelling for
    personal (non-business) purposes, from UNWTO's 'Inbound Tourism-Purpose'
    sheet (Personal / Total), nearest year to target_year within
    MAX_YEAR_GAP, with region-median imputation like the air share.

    Lambda is a share of personal travellers (Eurobarometer 499), so it is
    applied on top of this share rather than to all arrivals.
    """
    raw = pd.read_excel(workbook_path, sheet_name='Inbound Tourism-Purpose', header=2)
    raw['C.'] = raw['C.'].ffill()
    purpose = raw['Unnamed: 6'].where(raw['Unnamed: 6'] == 'Personal')
    purpose = purpose.fillna(raw['Unnamed: 5'].where(raw['Unnamed: 5'] == 'Total'))
    raw = raw.assign(purpose=purpose)
    raw = raw[raw['purpose'].notna()]

    year_cols = [c for c in raw.columns if isinstance(c, int)]
    tidy = raw.melt(id_vars=['C.', 'purpose'], value_vars=year_cols, var_name='year', value_name='arrivals')
    tidy = tidy.rename(columns={'C.': 'iso3_r250_id'})
    tidy['arrivals'] = pd.to_numeric(tidy['arrivals'], errors='coerce')
    tidy = ru.recode_unwto_ids(tidy.dropna(subset=['arrivals']))
    wide = tidy.pivot_table(index=['iso3_r250_id', 'year'], columns='purpose',
                            values='arrivals', aggfunc='sum').reset_index()
    wide = wide[wide['Personal'].notna() & (wide['Total'] > 0)]
    wide['personal_share'] = (wide['Personal'] / wide['Total']).clip(0.0, 1.0)

    best = ru.pick_nearest_year(wide, ['personal_share'], target_year)
    measured = best[best['year_distance'] <= MAX_YEAR_GAP].rename(columns={'year': 'personal_share_year'})
    out = ru.fill_by_region(measured[['iso3_r250_id', 'personal_share', 'personal_share_year']],
                            'personal_share', crosswalk_path, 'personal_share_imputed')
    return out[['iso3_r250_id', 'personal_share', 'personal_share_year', 'personal_share_imputed']]


def airfare_deflation_factor(cpi_path, survey_query_date, target_year=TARGET_YEAR):
    """
    Ratio of the target_year average to the survey-month value of the BLS
    CPI-U Airline Fares index (FRED series CUUR0000SETG01, U.S. city
    average, not seasonally adjusted). Falls back to the latest published
    month when the survey month isn't out yet.

    Returns (factor, month_used).
    """
    if not os.path.exists(cpi_path):
        raise FileNotFoundError(
            f"Missing airline-fare CPI: {cpi_path}. Download FRED series CUUR0000SETG01 "
            "(https://fred.stlouisfed.org/series/CUUR0000SETG01) as CSV into 0_inputs.")
    cpi = pd.read_csv(cpi_path)
    date_col = 'observation_date' if 'observation_date' in cpi.columns else 'DATE'
    cpi[date_col] = pd.to_datetime(cpi[date_col])
    cpi['CUUR0000SETG01'] = pd.to_numeric(cpi['CUUR0000SETG01'], errors='coerce')
    cpi = cpi.dropna(subset=['CUUR0000SETG01']).sort_values(date_col)
    cpi_target = cpi.loc[cpi[date_col].dt.year == target_year, 'CUUR0000SETG01'].mean()
    survey_month = pd.Timestamp(survey_query_date).to_period('M').to_timestamp()
    row = cpi.loc[cpi[date_col] == survey_month, 'CUUR0000SETG01']
    cpi_survey = row.iloc[0] if not row.empty else cpi['CUUR0000SETG01'].iloc[-1]
    month_used = survey_month if not row.empty else cpi[date_col].iloc[-1]
    return cpi_target / cpi_survey, month_used


def select_arrivals_year(arrivals, target_year=TARGET_YEAR):
    """
    Keep one year per country: the nearest to target_year (within
    MAX_YEAR_GAP) with any nonzero region-level arrivals, so all of a
    country's origin-region shares come from the same year.
    """
    totals = arrivals.groupby(['iso3_r250_id', 'year'])['arrivals'].sum().reset_index()
    totals = totals[totals['arrivals'] > 0]
    best = ru.pick_nearest_year(totals, [], target_year)
    year_choice = best.loc[best['year_distance'] <= MAX_YEAR_GAP, ['iso3_r250_id', 'year']]
    L.info(f"Usable arrivals-by-region data (within {MAX_YEAR_GAP} years of {target_year}) "
           f"for {len(year_choice)} countries")
    return arrivals.merge(year_choice.rename(columns={'year': 'year_used'}),
                          left_on=['iso3_r250_id', 'year'], right_on=['iso3_r250_id', 'year_used'])


def air_travel_value_by_country(arrivals_chosen, crosswalk_path, airfare, air_share):
    """
    air_travel_value = sum over origin regions of arrivals x air share x
    regional fare (origin region -> destination country's region).

    airfare: DataFrame with origin_region, destination_region,
             predicted_fare_2019_usd
    """
    crosswalk = pd.read_csv(crosswalk_path)[['iso3_r250_id', 'unwto_region']].rename(
        columns={'unwto_region': 'destination_region'})
    merged = arrivals_chosen.merge(crosswalk, on='iso3_r250_id', how='left')
    merged = merged.merge(airfare, on=['origin_region', 'destination_region'], how='left')
    merged = merged.merge(air_share[['iso3_r250_id', 'air_share']], on='iso3_r250_id', how='left')
    # "Other" origin/destination has no fare in the matrix (near-empty
    # residual UNWTO bucket, intentionally not scraped); those arrivals are
    # tracked but not priced.
    merged['priced_arrivals'] = np.where(merged['predicted_fare_2019_usd'].notna(), merged['arrivals'], 0.0)
    # only arrivals that actually fly are priced at an airfare
    merged['value'] = merged['arrivals'] * merged['air_share'] * merged['predicted_fare_2019_usd']

    country_df = merged.groupby('iso3_r250_id').agg(
        total_arrivals=('arrivals', 'sum'),
        priced_arrivals=('priced_arrivals', 'sum'),
        air_travel_value=('value', lambda s: s.sum(skipna=True)),
    ).reset_index()
    country_df['arrivals_coverage_share'] = np.where(
        country_df['total_arrivals'] > 0, country_df['priced_arrivals'] / country_df['total_arrivals'], np.nan)
    return country_df.merge(air_share, on='iso3_r250_id', how='left')


# ===========================================================================
# Country aggregation
# ===========================================================================

def combine_country_results(country_ids, daily_sites, tourist_sites, missing_cost_daily, missing_cost_tourist,
                            missing_overnights, accommodation, air_travel, personal_share):
    """
    Combine the four value components into one row per country.

    Missing inputs are NaN, not zero: no fuel cost -> NaN ground value; no
    overnights -> NaN tourist visits/value and accommodation; no arrivals ->
    NaN air travel. Accommodation and air travel are scaled by personal_share
    x NATURE_ATTRIBUTION_SHARE; unscaled values are kept as *_pre_attribution.
    grand_total_value sums whatever components are known.
    """
    daily = daily_sites.groupby('iso3_r250_id').agg(
        daily_visits=('visits', 'sum'), daily_value=('value', 'sum')).reset_index()
    tourist = tourist_sites.groupby('iso3_r250_id').agg(
        tourist_visits=('visits', 'sum'), tourist_value=('value', 'sum')).reset_index()

    # start from every country so ones with no high-quality site (zero
    # site-visit flows) still carry their accommodation and air values
    df = pd.DataFrame({'iso3_r250_id': country_ids}).drop_duplicates()
    df = df.merge(daily, on='iso3_r250_id', how='left').merge(tourist, on='iso3_r250_id', how='left')
    flow_cols = ['daily_visits', 'daily_value', 'tourist_visits', 'tourist_value']
    # Ring-band convolution round-trips leave float32 noise (~1e-9) that can
    # turn a true zero slightly negative; clamp.
    df[flow_cols] = df[flow_cols].fillna(0.0).clip(lower=0.0)

    df.loc[df['iso3_r250_id'].isin(missing_cost_daily), 'daily_value'] = np.nan
    df.loc[df['iso3_r250_id'].isin(missing_cost_tourist), 'tourist_value'] = np.nan
    df.loc[df['iso3_r250_id'].isin(missing_overnights), ['tourist_visits', 'tourist_value']] = np.nan

    df = df.merge(accommodation[['iso3_r250_id', 'accommodation_value', 'price_coverage_share']],
                  on='iso3_r250_id', how='left')
    df.loc[df['iso3_r250_id'].isin(missing_overnights), ['accommodation_value', 'price_coverage_share']] = np.nan

    df = df.merge(air_travel[['iso3_r250_id', 'total_arrivals', 'air_travel_value', 'arrivals_coverage_share']],
                  on='iso3_r250_id', how='left')
    no_arrivals_data = df['total_arrivals'].isna() | (df['total_arrivals'] == 0)
    df.loc[no_arrivals_data, ['air_travel_value', 'arrivals_coverage_share']] = np.nan
    df = df.drop(columns='total_arrivals')

    df = df.merge(personal_share[['iso3_r250_id', 'personal_share', 'personal_share_imputed']],
                  on='iso3_r250_id', how='left')
    attribution = df['personal_share'] * NATURE_ATTRIBUTION_SHARE
    df['accommodation_value_pre_attribution'] = df['accommodation_value']
    df['air_travel_value_pre_attribution'] = df['air_travel_value']
    df['accommodation_value'] = df['accommodation_value'] * attribution
    df['air_travel_value'] = df['air_travel_value'] * attribution

    components = ['daily_value', 'tourist_value', 'accommodation_value', 'air_travel_value']
    df['n_components_missing'] = df[components].isna().sum(axis=1)
    df['grand_total_value'] = df[components].sum(axis=1, min_count=1)
    return df
