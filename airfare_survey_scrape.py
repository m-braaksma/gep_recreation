"""
airfare_survey_scrape.py

Builds the pinned region-to-region airfare survey used by the recreation
pipeline (0_inputs/airfare_2019_regions.csv). Run this deliberately, not as
part of run_recreation.py: it makes live requests to a third-party flight
search API, so results depend on the date it is run and on that API's
availability.

Method:
  1. Scrape current round-trip fares (flightlist.io public JSON endpoint,
     no API key) for every ordered pair among 3 representative hub airports
     per UNWTO region (18 hubs total; "Other" is skipped - see
     unwto_region_crosswalk.csv, it is a near-empty residual bucket real
     countries essentially never fall into).
  2. Average the 5 cheapest quotes per hub-pair, then average all hub-pairs
     belonging to the same region-pair.
  3. Deflate today's fare to a 2019-equivalent fare using the ratio of the
     BLS CPI-U Airline Fares index (CUUR0000SETG01) 2019 average to its
     value in the survey month. The pipeline recomputes this deflation from
     survey_query_date, so the CPI file can be updated later.

Usage:
    python airfare_survey_scrape.py <data_dir> [output_dir]
"""

import os
import sys
import itertools
import time
import json
import datetime
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd

import recreation_functions as rf

# 3 hubs per UNWTO region, chosen for sub-regional geographic spread and
# consistent with unwto_region_crosswalk.csv (e.g. Egypt hubs excluded from
# Africa since Egypt is UNWTO Middle East). "Other" is intentionally omitted.
HUBS = {
    'Africa': ['JNB', 'NBO', 'LOS'],                      # Johannesburg, Nairobi, Lagos
    'Americas': ['ATL', 'GRU', 'MEX'],                    # Atlanta, Sao Paulo, Mexico City
    'East Asia and the Pacific': ['HND', 'SIN', 'SYD'],   # Tokyo, Singapore, Sydney
    'Europe': ['LHR', 'CDG', 'IST'],                      # London, Paris, Istanbul
    'Middle East': ['DXB', 'DOH', 'RUH'],                 # Dubai, Doha, Riyadh
    'South Asia': ['DEL', 'DAC', 'CMB'],                  # Delhi, Dhaka, Colombo
}

REQUEST_DELAY_SECONDS = 0.4
N_CHEAPEST_TO_AVERAGE = 5


def flightlist_query_url(origin_iata, destination_iata, outbound_date, return_date):
    query = {
        'fly_from': f'airport:{origin_iata}', 'fly_to': f'airport:{destination_iata}',
        'date_from': outbound_date.strftime('%d/%m/%Y'), 'date_to': outbound_date.strftime('%d/%m/%Y'),
        'return_from': return_date.strftime('%d/%m/%Y'), 'return_to': return_date.strftime('%d/%m/%Y'),
        'adults': '1', 'children': '0', 'infants': '0', 'selected_cabins': 'M', 'curr': 'USD',
        'limit': '20', 'sort': 'price', 'max_stopovers': '10', 'flight_type': 'return',
        'adult_hand_bag': '0', 'adult_hold_bag': '0',
    }
    return 'https://www.flightlist.io/api/search.php?' + urllib.parse.urlencode(query)


def fetch_cheapest_fares(origin_iata, destination_iata, outbound_date, return_date):
    url = flightlist_query_url(origin_iata, destination_iata, outbound_date, return_date)
    request = urllib.request.Request(url, headers={
        'User-Agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'),
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.flightlist.io/',
    })
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode('utf-8'))
    except Exception as error:
        print(f"  {origin_iata}->{destination_iata}: request failed ({error})")
        return []
    prices = sorted(f['price'] for f in payload.get('data', []) if f.get('price') is not None)
    return prices[:N_CHEAPEST_TO_AVERAGE]


def build_airfare_regions(data_dir, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'gep_recreation', '0_inputs')
    output_csv = os.path.join(output_dir, 'airfare_2019_regions.csv')

    outbound_date = datetime.date.today() + datetime.timedelta(days=21)
    outbound_date += datetime.timedelta(days=(4 - outbound_date.weekday()) % 7)  # next Friday
    return_date = outbound_date + datetime.timedelta(days=7)
    print(f"Survey dates: {outbound_date} -> {return_date}")

    hub_to_region = {iata: region for region, iatas in HUBS.items() for iata in iatas}
    all_hubs = list(hub_to_region.keys())
    pairs = [(o, d) for o, d in itertools.product(all_hubs, all_hubs) if o != d]
    print(f"Querying {len(pairs)} hub-pairs across {len(HUBS)} regions "
          f"({len(all_hubs)} hubs, {N_CHEAPEST_TO_AVERAGE}-cheapest average each)...")

    rows = []
    for i, (origin, destination) in enumerate(pairs):
        prices = fetch_cheapest_fares(origin, destination, outbound_date, return_date)
        avg_price = float(np.mean(prices)) if prices else np.nan
        rows.append({
            'origin_iata': origin, 'destination_iata': destination,
            'origin_region': hub_to_region[origin], 'destination_region': hub_to_region[destination],
            'n_quotes': len(prices), 'average_fare_usd': avg_price,
        })
        if (i + 1) % 20 == 0 or i == len(pairs) - 1:
            print(f"  {i + 1}/{len(pairs)} pairs done")
        time.sleep(REQUEST_DELAY_SECONDS)

    hub_pairs = pd.DataFrame(rows)
    hub_pairs.to_csv(os.path.join(output_dir, 'airfare_hub_pairs_raw.csv'), index=False)

    missing = hub_pairs['average_fare_usd'].isna().sum()
    if missing:
        print(f"Warning: {missing} of {len(hub_pairs)} hub-pairs returned no usable quote")

    region_pairs = hub_pairs.groupby(['origin_region', 'destination_region'])['average_fare_usd'].mean().reset_index()
    region_pairs = region_pairs.rename(columns={'average_fare_usd': 'average_fare_usd_current'})

    survey_query_date = datetime.date.today()
    cpi_path = os.path.join(data_dir, 'gep_recreation', '0_inputs', 'CUUR0000SETG01.csv')
    deflation_factor, cpi_month = rf.airfare_deflation_factor(cpi_path, survey_query_date)
    print(f"Airline-fare CPI deflation factor = {deflation_factor:.4f} (month used: {cpi_month.date()})")

    region_pairs['survey_query_date'] = survey_query_date.isoformat()
    region_pairs['survey_outbound_date'] = outbound_date.isoformat()
    region_pairs['deflation_factor'] = deflation_factor
    region_pairs['predicted_fare_2019_usd'] = region_pairs['average_fare_usd_current'] * deflation_factor
    region_pairs['year'] = rf.TARGET_YEAR

    region_pairs.to_csv(output_csv, index=False)
    print(f"Saved region-pair 2019 airfare matrix to {output_csv}")
    print(region_pairs.sort_values(['origin_region', 'destination_region']).to_string(index=False))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    build_airfare_regions(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
