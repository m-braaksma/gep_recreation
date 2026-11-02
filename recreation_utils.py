"""
recreation_utils.py

Generic raster, I/O and data-wrangling helpers shared by recreation_tasks.py
and recreation_functions.py. Nothing here encodes a modelling choice.
"""

import os
import numpy as np
import pandas as pd
import pygeoprocessing
from osgeo import gdal, osr
import hazelbean as hb

L = hb.get_logger('recreation_utils')


# ===========================================================================
# Input checks
# ===========================================================================

def require_files(paths_by_name, context):
    """Raise one FileNotFoundError listing every missing input."""
    missing = [f"{k}: {v}" for k, v in paths_by_name.items() if not os.path.exists(v)]
    if missing:
        L.error(f"Missing input files for {context}:\n" + "\n".join(missing))
        raise FileNotFoundError(f"Missing input files for {context}:\n" + "\n".join(missing))


def outputs_exist(paths):
    paths = paths.values() if isinstance(paths, dict) else paths
    return all(os.path.exists(v) for v in paths)


def assert_rasters_aligned(reference_path, other_paths_by_name, tolerance=1e-6):
    """
    Raise a clear error if any raster doesn't share the reference raster's
    pixel grid (raster_size, pixel_size, and origin/bounding_box). The flow
    model reuses one raster's (xoff, yoff, wx, wy) window to read windows
    out of several other rasters on the assumption they share a grid.
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


def check_country_raster(country_raster_path):
    """
    Guard against a known pygeoprocessing rasterize failure mode in this
    environment (all-zero output with no raised error).
    """
    country_nodata = pygeoprocessing.get_raster_info(country_raster_path)['nodata'][0]
    ds = gdal.Open(country_raster_path)
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    valid = (arr != country_nodata) & (arr > 0) if country_nodata is not None else (arr > 0)
    n_valid = int(valid.sum())
    n_unique = len(np.unique(arr[valid])) if n_valid else 0
    if n_valid == 0 or n_unique < 2:
        raise ValueError(
            f"Country-ID raster {country_raster_path} looks broken: "
            f"{n_valid} valid pixels, {n_unique} distinct country codes. "
            "Expected ~hundreds of millions of valid pixels across ~250 "
            "countries. This matches a known pygeoprocessing rasterize "
            "wrapper failure mode seen in this environment (all-zero "
            "output with no raised error) - try gdal.Rasterize() directly.")
    L.info(f"Country-ID raster sanity check passed: {n_valid:,} valid pixels, {n_unique} distinct countries")


def log_country_coverage(country_id_path, covered_ids, label, consequence, group_id_path=None):
    """
    Stream over the country-id raster and report which countries present in
    it have an entry in covered_ids. With group_id_path, coverage is tested
    on each pixel's param_group instead of its country id.

    Returns (covered, uncovered) sets of iso3_r250_id. A country with even
    one covered pixel counts as covered.
    """
    country_nodata = pygeoprocessing.get_raster_info(country_id_path)['nodata'][0]
    valid_ids = np.array(sorted(covered_ids))

    country_ds = gdal.Open(country_id_path); country_band = country_ds.GetRasterBand(1)
    group_ds = gdal.Open(group_id_path) if group_id_path else None
    group_band = group_ds.GetRasterBand(1) if group_ds else None

    covered, uncovered = set(), set()
    for block_info in pygeoprocessing.iterblocks((country_id_path, 1), offset_only=True):
        xoff, yoff = block_info['xoff'], block_info['yoff']
        wx, wy = block_info['win_xsize'], block_info['win_ysize']
        country_block = country_band.ReadAsArray(xoff, yoff, wx, wy)
        valid = (country_block != country_nodata) & (country_block > 0) if country_nodata is not None \
            else (country_block > 0)
        if not np.any(valid):
            continue
        test_block = group_band.ReadAsArray(xoff, yoff, wx, wy) if group_band else country_block
        has = np.isin(test_block[valid], valid_ids)
        covered.update(np.unique(country_block[valid][has]).tolist())
        uncovered.update(np.unique(country_block[valid][~has]).tolist())
    uncovered -= covered
    country_band = country_ds = group_band = group_ds = None

    L.info(f"{label} coverage: {len(covered)} countries covered, {len(uncovered)} countries "
           f"present in the country_id raster with no {label} data ({consequence})")
    if uncovered:
        L.warning(f"iso3_r250_id values with no {label} data: {sorted(uncovered)}")
    return covered, uncovered


# ===========================================================================
# Rasterizing and raster I/O
# ===========================================================================

def rasterize_points_count(vector_path, ref_raster_path, output_path):
    """Count of points per pixel on the reference raster's grid."""
    if os.path.exists(output_path):
        L.info(f"Point-count raster already exists: {output_path}")
        return output_path
    ref_info = pygeoprocessing.get_raster_info(ref_raster_path)
    pygeoprocessing.geoprocessing.create_raster_from_bounding_box(
        target_bounding_box=ref_info['bounding_box'], target_raster_path=output_path,
        target_pixel_size=ref_info['pixel_size'], target_pixel_type=gdal.GDT_Int16,
        target_srs_wkt=ref_info['projection_wkt'], target_nodata=-1, fill_value=0)
    pygeoprocessing.geoprocessing.rasterize(
        vector_path=vector_path, target_raster_path=output_path,
        burn_values=[1], option_list=['ALL_TOUCHED=TRUE', 'MERGE_ALG=ADD'])
    return output_path


def rasterize_attribute(vector_path, ref_raster_path, output_path, column):
    """Burn a vector attribute onto the reference raster's grid."""
    if os.path.exists(output_path):
        L.info(f"Rasterized {column} already exists: {output_path}")
        return output_path
    ref_info = pygeoprocessing.get_raster_info(ref_raster_path)
    pygeoprocessing.geoprocessing.create_raster_from_bounding_box(
        target_bounding_box=ref_info['bounding_box'], target_raster_path=output_path,
        target_pixel_size=ref_info['pixel_size'], target_pixel_type=ref_info['datatype'],
        target_srs_wkt=ref_info['projection_wkt'], target_nodata=ref_info['nodata'][0], fill_value=0)
    pygeoprocessing.geoprocessing.rasterize(
        vector_path=vector_path, target_raster_path=output_path,
        burn_values=None, option_list=['ALL_TOUCHED=TRUE', f'ATTRIBUTE={column}'])
    return output_path


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
    linear_unit_name = srs.GetLinearUnitsName()
    if linear_unit_name and 'kilomet' in linear_unit_name.lower():
        return pixel_size_native
    return pixel_size_native / 1000.0


def write_array_as_raster(array, pixel_size_km, output_path, nodata=-9999):
    """Write a plain numpy array to a small local GeoTIFF -- used both for
    kernels and for per-tile rasters fed into pygeoprocessing.convolve_2d,
    which requires file-based inputs. The geotransform is a local
    placeholder: convolve_2d aligns by pixel count, so only the pixel size
    matters, not real-world placement."""
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


def read_raster_array(path):
    ds = gdal.Open(path)
    arr = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    return arr


def read_window(band, xoff, yoff, wx, wy):
    return band.ReadAsArray(xoff, yoff, wx, wy)


def accumulate_window(band, xoff, yoff, window_contribution):
    """Read-modify-write add: buffered windows can overlap (population near
    a tile edge contributes to sites just across it), so a later window's
    contribution must be added to, not overwrite, an earlier one's."""
    existing = band.ReadAsArray(xoff, yoff, window_contribution.shape[1], window_contribution.shape[0])
    band.WriteArray(existing + window_contribution, xoff, yoff)


def scan_group_bboxes(group_id_path, group_nodata=-1):
    """
    One streaming pass over the group_id raster to find each group's pixel
    bounding box. Returns dict: group_id -> [row_min, row_max, col_min,
    col_max] (inclusive).
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


def iter_core_tiles(r0, r1, c0, c1, tile_rows, tile_cols):
    """
    Split an inclusive [r0,r1] x [c0,c1] pixel bbox into non-overlapping
    core tiles. Buffering each tile is the caller's job; this only assigns
    which tile owns each pixel's demand, so overlapping buffered read
    windows never double-count.
    """
    for tr0 in range(r0, r1 + 1, tile_rows):
        tr1 = min(tr0 + tile_rows - 1, r1)
        for tc0 in range(c0, c1 + 1, tile_cols):
            tc1 = min(tc0 + tile_cols - 1, c1)
            yield tr0, tr1, tc0, tc1


# ===========================================================================
# UNWTO workbook helpers
# ===========================================================================

# UNWTO uses some legacy/non-standard M49 codes that either don't exist in
# the iso3_r250_id raster (e.g. Germany is 280, the old West Germany code, vs
# 276) or exist but mean a different place (UNWTO 535 is Curacao, r250 535 is
# Bonaire/Sint Eustatius/Saba). Checked by name for every shared code.
# None = drop: Serbia and Montenegro (891) is superseded by separately
# reported Serbia (688) and Montenegro (499).
UNWTO_TO_R250_ID = {
    280: 276,  # Germany
    230: 231,  # Ethiopia
    886: 887,  # Yemen
    736: 729,  # Sudan
    535: 531,  # Curacao
    534: 535,  # Bonaire -> Bonaire, Sint Eustatius and Saba
    461: 535,  # Saba -> Bonaire, Sint Eustatius and Saba
    658: 535,  # Sint Eustatius -> Bonaire, Sint Eustatius and Saba
    663: 534,  # Sint Maarten (Dutch part)
    891: None,  # Serbia and Montenegro
}


def recode_unwto_ids(df, col='iso3_r250_id'):
    recoded = df[col].astype(int).map(lambda c: UNWTO_TO_R250_ID.get(c, c))
    df = df.assign(**{col: recoded})
    df = df[df[col].notna()].copy()
    df[col] = df[col].astype(int)
    return df


def read_unwto_sheet(workbook_path, sheet_name):
    """Read a UNWTO sheet starting at its first country row (Afghanistan)."""
    head = pd.read_excel(workbook_path, sheet_name=sheet_name)
    start_idx = head[head.iloc[:, 3] == 'AFGHANISTAN'].index[0]
    return pd.read_excel(workbook_path, sheet_name=sheet_name, skiprows=start_idx)


def pick_nearest_year(df, value_cols, target_year, id_col='iso3_r250_id'):
    """
    One row per id: the year closest to target_year (ties go to the later
    year), with a year_distance column. Callers apply MAX_YEAR_GAP.
    """
    rows = []
    for cid, sub in df.groupby(id_col):
        sub = sub.assign(year_distance=(sub['year'] - target_year).abs())
        best = sub.sort_values(['year_distance', 'year'], ascending=[True, False]).iloc[0]
        rows.append({id_col: cid, 'year': int(best['year']), 'year_distance': int(best['year_distance']),
                     **{c: best[c] for c in value_cols}})
    return pd.DataFrame(rows, columns=[id_col, 'year', 'year_distance'] + list(value_cols))


def fill_by_region(df, value_col, crosswalk_path, flag_col):
    """
    Left-join df onto every country in the UNWTO region crosswalk and fill
    missing value_col with the region median, then the global median.
    """
    crosswalk = pd.read_csv(crosswalk_path)[['iso3_r250_id', 'unwto_region']]
    out = crosswalk.merge(df, on='iso3_r250_id', how='left')
    region_median = out.groupby('unwto_region')[value_col].median()
    global_median = out[value_col].median()
    out[flag_col] = out[value_col].isna()
    out[value_col] = out[value_col].fillna(out['unwto_region'].map(region_median)).fillna(global_median)
    return out.drop(columns='unwto_region')


# --- GEP results table ------------------------------------------------------

GEP_COMPONENTS = {
    'daily_value': ('Daily recreation', 'ᵃ'),
    'tourist_value': ('Tourist recreation', 'ᵇ'),
    'accommodation_value': ('Accommodation', 'ᶜ'),
    'air_travel_value': ('Air travel', 'ᵈ'),
}
GEP_FOOTNOTES = {
    'ᵃ': 'Daily recreation not estimated (no fuel cost data).',
    'ᵇ': 'Tourist recreation not estimated (no UNWTO overnights or fuel cost data).',
    'ᶜ': 'Accommodation not estimated (no UNWTO overnights data).',
    'ᵈ': 'Air travel not estimated (no UNWTO arrivals data).',
}


def format_gep_table(df, units=1e6):
    """
    Reshape country results into a GEP-style table for pasting into the
    combined results: one row per country in millions of 2019 USD, a global
    total row, and the footnote legend below it. A missing component is
    written as 'NA' with a superscript footnote marker, not zero.
    """
    df = df.sort_values('iso3_r250_label').reset_index(drop=True)
    out = pd.DataFrame({'Country (ISO3)': list(df['iso3_r250_label']) + ['Total']})
    for col, (label, marker) in GEP_COMPONENTS.items():
        values = df[col] / units
        cells = [f'NA{marker}' if pd.isna(v) else f'{v:.2f}' for v in values]
        out[f'{label} (million 2019 USD)'] = cells + [f'{values.sum():.2f}']
    total = df['grand_total_value'] / units
    out['Total (million 2019 USD)'] = [
        'NA' if pd.isna(v) else f'{v:.2f}' for v in total] + [f'{total.sum():.2f}']
    legend = pd.DataFrame({'Country (ISO3)': [f'{m} {text}' for m, text in GEP_FOOTNOTES.items()]})
    return pd.concat([out, legend], ignore_index=True)
