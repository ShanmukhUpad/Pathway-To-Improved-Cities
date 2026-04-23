"""
city_config.py
--------------
<<<<<<< Updated upstream
Central configuration for all supported cities. Each city defines its
Socrata open-data portal URL, dataset IDs, column mappings, GeoJSON
boundaries, map centre, and region metadata.

The downstream pipeline normalises every city's raw data into a single
canonical schema at fetch time so that ML models and visualisation code
never see portal-specific column names.
"""

import os

_SRC = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_SRC, "data")


def _data_dir(city_key: str) -> str:
    return os.path.join(_DATA, city_key)


# ──────────────────────────────────────────────
# Canonical column names (after normalisation)
# ──────────────────────────────────────────────
# Crime pivot CSV:  Community Area, Year, Month, <CRIME_TYPE_1>, ...
# Crash CSV:        CRASH_DATE, WEATHER_CONDITION, LIGHTING_CONDITION,
#                   ROADWAY_SURFACE_COND, ROAD_DEFECT, POSTED_SPEED_LIMIT,
#                   LATITUDE, LONGITUDE, FIRST_CRASH_TYPE, CRASH_TYPE,
#                   DAMAGE, NUM_UNITS, HIT_AND_RUN_I, INTERSECTION_RELATED_I,
#                   ALIGNMENT, TRAFFICWAY_TYPE, LANE_CNT,
#                   TRAFFIC_CONTROL_DEVICE, DEVICE_CONDITION,
#                   BEAT_OF_OCCURRENCE

CITIES = {
    # ════════════════════════════════════════════
    # CHICAGO
    # ════════════════════════════════════════════
    "chicago": {
        "display_name": "Chicago",
        "portal_url": "https://data.cityofchicago.org/resource",
        "app_token_env": "CHICAGO_DATA_PORTAL_TOKEN",

        # Dataset IDs
        "crime_dataset_id": "ijzp-q8t2",
        "crash_dataset_id": "85ca-t3if",

        # Crime API ─ $select / $where fragments + column mapping
        "crime_select": "community_area,year,date,primary_type",
        "crime_where_tpl": "year >= '{start_year}'",
        "crime_col_map": {
            "date": "date",
            "type": "primary_type",
            "area": "community_area",
            "year": "year",
        },

        # Crash API
        "crash_select": (
            "crash_date,weather_condition,lighting_condition,"
            "roadway_surface_cond,road_defect,alignment,trafficway_type,"
            "lane_cnt,posted_speed_limit,traffic_control_device,"
            "device_condition,intersection_related_i,first_crash_type,"
            "crash_type,damage,num_units,hit_and_run_i,"
            "latitude,longitude,beat_of_occurrence"
        ),
        "crash_where_tpl": "crash_date >= '{start_date}'",
        "crash_col_map": {
            "CRASH_DATE": "crash_date",
            "WEATHER_CONDITION": "weather_condition",
            "LIGHTING_CONDITION": "lighting_condition",
            "ROADWAY_SURFACE_COND": "roadway_surface_cond",
            "ROAD_DEFECT": "road_defect",
            "ALIGNMENT": "alignment",
            "TRAFFICWAY_TYPE": "trafficway_type",
            "LANE_CNT": "lane_cnt",
            "POSTED_SPEED_LIMIT": "posted_speed_limit",
            "TRAFFIC_CONTROL_DEVICE": "traffic_control_device",
            "DEVICE_CONDITION": "device_condition",
            "INTERSECTION_RELATED_I": "intersection_related_i",
            "FIRST_CRASH_TYPE": "first_crash_type",
            "CRASH_TYPE": "crash_type",
            "DAMAGE": "damage",
            "NUM_UNITS": "num_units",
            "HIT_AND_RUN_I": "hit_and_run_i",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
            "BEAT_OF_OCCURRENCE": "beat_of_occurrence",
        },

        # Geography
        "geojson_url": "https://raw.githubusercontent.com/RandomFractals/ChicagoCrimes/master/data/chicago-community-areas.geojson",
        "geojson_id_field": "area_num_1",
        "geojson_name_field": "community",
        "center": {"lat": 41.8781, "lon": -87.6298},
        "lat_bounds": (41.6, 42.1),
        "zoom": 9,

        # Region terminology
        "region_label": "Community Area",
        "region_names": {
            1: "Rogers Park", 2: "West Ridge", 3: "Uptown", 4: "Lincoln Square",
            5: "North Center", 6: "Lake View", 7: "Lincoln Park", 8: "Near North Side",
            9: "Edison Park", 10: "Norwood Park", 11: "Jefferson Park", 12: "Forest Glen",
            13: "North Park", 14: "Albany Park", 15: "Portage Park", 16: "Irving Park",
            17: "Dunning", 18: "Montclare", 19: "Belmont Cragin", 20: "Hermosa",
            21: "Avondale", 22: "Logan Square", 23: "Humboldt Park", 24: "West Town",
            25: "Austin", 26: "West Garfield Park", 27: "East Garfield Park", 28: "Near West Side",
            29: "North Lawndale", 30: "South Lawndale", 31: "Lower West Side", 32: "Loop",
            33: "Near South Side", 34: "Armour Square", 35: "Douglas", 36: "Oakland",
            37: "Fuller Park", 38: "Grand Boulevard", 39: "Kenwood", 40: "Washington Park",
            41: "Hyde Park", 42: "Woodlawn", 43: "South Shore", 44: "Chatham",
            45: "Avalon Park", 46: "South Chicago", 47: "Burnside", 48: "Calumet Heights",
            49: "Roseland", 50: "Pullman", 51: "South Deering", 52: "East Side",
            53: "West Pullman", 54: "Riverdale", 55: "Hegewisch", 56: "Garfield Ridge",
            57: "Archer Heights", 58: "Brighton Park", 59: "McKinley Park", 60: "Bridgeport",
            61: "New City", 62: "West Elsdon", 63: "Gage Park", 64: "Clearing",
            65: "West Lawn", 66: "Chicago Lawn", 67: "West Englewood", 68: "Englewood",
            69: "Greater Grand Crossing", 70: "Ashburn", 71: "Auburn Gresham", 72: "Beverly",
            73: "Washington Heights", 74: "Mount Greenwood", 75: "Morgan Park",
            76: "O'Hare", 77: "Edgewater",
        },

        # Ward GeoJSON (transportation access tab)
        "ward_geojson_url": "https://data.cityofchicago.org/resource/p293-wvbd.geojson",

        # Socioeconomic
        "census_csv": "censusChicago.csv",
        "census_area_col": "Community Area Number",
        "census_name_col": "COMMUNITY AREA NAME",
        "census_hardship_col": "HARDSHIP INDEX",
        "census_feature_cols": [
            "PERCENT OF HOUSING CROWDED",
            "PERCENT HOUSEHOLDS BELOW POVERTY",
            "PERCENT AGED 16+ UNEMPLOYED",
            "PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA",
            "PERCENT AGED UNDER 18 OR OVER 64",
            "PER CAPITA INCOME",
        ],
        "census_feature_short": [
            "Housing Crowded", "Below Poverty", "Unemployed 16+",
            "No HS Diploma", "Under 18/Over 64", "Per Capita Income",
        ],

        # Transportation CSVs (may not exist)
        "transit_csvs": {
            "bus": "bus_stops_clean.csv",
            "divvy": "divvy_bicycle_clean.csv",
            "bike": "bike_routes_clean.csv",
        },
    },

    # ════════════════════════════════════════════
    # NEW YORK CITY
    # Verified: data.cityofnewyork.us
    # Crime:  5uac-w243  (NYPD Complaint Data Current YTD)
    # Crash:  h9gi-nx95  (Motor Vehicle Collisions - Crashes)
    # ════════════════════════════════════════════
    "new_york": {
        "display_name": "New York City",
        "portal_url": "https://data.cityofnewyork.us/resource",
        "app_token_env": "NYC_DATA_PORTAL_TOKEN",

        "crime_dataset_id": "5uac-w243",
        "crash_dataset_id": "h9gi-nx95",

        # Verified columns: cmplnt_fr_dt, ofns_desc, addr_pct_cd
        "crime_select": "addr_pct_cd,cmplnt_fr_dt,ofns_desc",
        "crime_where_tpl": "cmplnt_fr_dt >= '{start_year}-01-01T00:00:00'",
        "crime_col_map": {
            "date": "cmplnt_fr_dt",
            "type": "ofns_desc",
            "area": "addr_pct_cd",
        },

        # Verified columns: crash_date, latitude, longitude,
        #   contributing_factor_vehicle_1, vehicle_type_code1,
        #   number_of_persons_injured, borough
        "crash_select": (
            "crash_date,contributing_factor_vehicle_1,"
            "vehicle_type_code1,number_of_persons_injured,"
            "number_of_persons_killed,latitude,longitude,borough"
        ),
        "crash_where_tpl": "crash_date >= '{start_date}'",
        "crash_col_map": {
            "CRASH_DATE":            "crash_date",
            "WEATHER_CONDITION":     "contributing_factor_vehicle_1",
            "LIGHTING_CONDITION":    None,
            "ROADWAY_SURFACE_COND":  None,
            "ROAD_DEFECT":           None,
            "ALIGNMENT":             None,
            "TRAFFICWAY_TYPE":       "vehicle_type_code1",
            "LANE_CNT":              None,
            "POSTED_SPEED_LIMIT":    None,
            "TRAFFIC_CONTROL_DEVICE":None,
            "DEVICE_CONDITION":      None,
            "INTERSECTION_RELATED_I":None,
            "FIRST_CRASH_TYPE":      "contributing_factor_vehicle_1",
            "CRASH_TYPE":            "contributing_factor_vehicle_1",
            "DAMAGE":                None,
            "NUM_UNITS":             "number_of_persons_injured",
            "HIT_AND_RUN_I":         None,
            "LATITUDE":              "latitude",
            "LONGITUDE":             "longitude",
            "BEAT_OF_OCCURRENCE":    "borough",
        },

        "geojson_url": "https://data.cityofnewyork.us/api/geospatial/kmub-pusk?method=export&type=GeoJSON",
        "geojson_id_field": "precinct",
        "geojson_name_field": "precinct",
        "center": {"lat": 40.7128, "lon": -74.0060},
        "lat_bounds": (40.49, 40.92),
        "zoom": 9,

        "region_label": "Precinct",
        "region_names": {i: f"Precinct {i}" for i in range(1, 78)},

        "ward_geojson_url": None,

        # Census API — census tracts in NYC's 5 boroughs
        # State 36, counties 005 (Bronx) 047 (Brooklyn) 061 (Manhattan)
        #                      081 (Queens) 085 (Staten Island)
        "census_state": "36",
        "census_county": "005,047,061,081,085",

        "census_csv": "census.csv",
        "census_area_col": "GEOID",
        "census_name_col": "NAME",
        "census_hardship_col": "HARDSHIP INDEX",
        "census_feature_cols": [
            "PERCENT OF HOUSING CROWDED",
            "PERCENT HOUSEHOLDS BELOW POVERTY",
            "PERCENT AGED 16+ UNEMPLOYED",
            "PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA",
            "PERCENT AGED UNDER 18 OR OVER 64",
            "PER CAPITA INCOME",
        ],
        "census_feature_short": [
            "Housing Crowded", "Below Poverty", "Unemployed 16+",
            "No HS Diploma", "Under 18/Over 64", "Per Capita Income",
        ],

        "transit_csvs": {},
    },

    # ════════════════════════════════════════════
    # LOS ANGELES
    # Verified: data.lacity.org
    # Crime:  2nrs-mtv8  (Crime Data from 2020 to 2024)
    # Crash:  d5tf-ez2w  (Traffic Collision Data from 2010 to Present)
    #   NOTE: crash dataset has no explicit lat/lon columns —
    #         coordinates are embedded in the `location_1` geo field.
    #         Density maps will fall back gracefully.
    # ════════════════════════════════════════════
    "los_angeles": {
        "display_name": "Los Angeles",
        "portal_url": "https://data.lacity.org/resource",
        "app_token_env": "LA_DATA_PORTAL_TOKEN",

        "crime_dataset_id": "2nrs-mtv8",
        "crash_dataset_id": "d5tf-ez2w",

        # Verified columns: date_occ, crm_cd_desc, area_name
        # NOTE: lat/lon columns are named `lat` and `lon` (not latitude/longitude)
        "crime_select": "area_name,date_occ,crm_cd_desc,lat,lon",
        "crime_where_tpl": "date_occ >= '{start_year}-01-01T00:00:00'",
        "crime_col_map": {
            "date": "date_occ",
            "type": "crm_cd_desc",
            "area": "area_name",
        },

        # Verified columns: date_occ, area_name, crm_cd, crm_cd_desc,
        #   mocodes, premis_desc, location (address string), location_1 (geo)
        # No explicit latitude/longitude columns in this dataset.
        "crash_select": "date_occ,area_name,crm_cd_desc,mocodes,premis_desc",
        "crash_where_tpl": "date_occ >= '{start_date}'",
        "crash_col_map": {
            "CRASH_DATE":            "date_occ",
            "WEATHER_CONDITION":     None,
            "LIGHTING_CONDITION":    None,
            "ROADWAY_SURFACE_COND":  None,
            "ROAD_DEFECT":           None,
            "ALIGNMENT":             None,
            "TRAFFICWAY_TYPE":       None,
            "LANE_CNT":              None,
            "POSTED_SPEED_LIMIT":    None,
            "TRAFFIC_CONTROL_DEVICE":None,
            "DEVICE_CONDITION":      None,
            "INTERSECTION_RELATED_I":None,
            "FIRST_CRASH_TYPE":      "crm_cd_desc",
            "CRASH_TYPE":            "crm_cd_desc",
            "DAMAGE":                None,
            "NUM_UNITS":             None,
            "HIT_AND_RUN_I":         None,
            "LATITUDE":              None,   # not available
            "LONGITUDE":             None,   # not available
            "BEAT_OF_OCCURRENCE":    "area_name",
        },

        # LAPD Division boundaries GeoJSON
        "geojson_url": "https://opendata.arcgis.com/datasets/031d488e158144d0b3aecaa9c888b7b3_0.geojson",
        "geojson_id_field": "APREC",
        "geojson_name_field": "PREC",
        "center": {"lat": 34.0522, "lon": -118.2437},
        "lat_bounds": (33.7, 34.4),
        "zoom": 9,

        "region_label": "LAPD Area",
        "region_names": {
            "Central": "Central", "Rampart": "Rampart", "Southwest": "Southwest",
            "Hollenbeck": "Hollenbeck", "Harbor": "Harbor", "Hollywood": "Hollywood",
            "Wilshire": "Wilshire", "West LA": "West LA", "Van Nuys": "Van Nuys",
            "West Valley": "West Valley", "Northeast": "Northeast",
            "77th Street": "77th Street", "Newton": "Newton", "Pacific": "Pacific",
            "N Hollywood": "N Hollywood", "Foothill": "Foothill",
            "Devonshire": "Devonshire", "Southeast": "Southeast",
            "Mission": "Mission", "Olympic": "Olympic", "Topanga": "Topanga",
        },

        "ward_geojson_url": None,

        # Census API — census tracts in LA County (state 06, county 037)
        "census_state": "06",
        "census_county": "037",

        "census_csv": "census.csv",
        "census_area_col": "GEOID",
        "census_name_col": "NAME",
        "census_hardship_col": "HARDSHIP INDEX",
        "census_feature_cols": [
            "PERCENT OF HOUSING CROWDED",
            "PERCENT HOUSEHOLDS BELOW POVERTY",
            "PERCENT AGED 16+ UNEMPLOYED",
            "PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA",
            "PERCENT AGED UNDER 18 OR OVER 64",
            "PER CAPITA INCOME",
        ],
        "census_feature_short": [
            "Housing Crowded", "Below Poverty", "Unemployed 16+",
            "No HS Diploma", "Under 18/Over 64", "Per Capita Income",
        ],

        "transit_csvs": {},
    },

    # ════════════════════════════════════════════
    # SAN FRANCISCO
    # Verified: data.sfgov.org
    # Crime:  wg3w-h783  (Police Department Incident Reports 2018-Present)
    # Crash:  ubvf-ztfx  (Traffic Crashes Resulting in Injury)
    # ════════════════════════════════════════════
    "san_francisco": {
        "display_name": "San Francisco",
        "portal_url": "https://data.sfgov.org/resource",
        "app_token_env": "SF_DATA_PORTAL_TOKEN",

        "crime_dataset_id": "wg3w-h783",
        "crash_dataset_id": "ubvf-ztfx",

        # Verified columns: incident_datetime, incident_category, analysis_neighborhood
        "crime_select": "analysis_neighborhood,incident_datetime,incident_category",
        "crime_where_tpl": "incident_datetime >= '{start_year}-01-01T00:00:00'",
        "crime_col_map": {
            "date": "incident_datetime",
            "type": "incident_category",
            "area": "analysis_neighborhood",
        },

        # Verified columns: collision_datetime, tb_latitude, tb_longitude,
        #   weather_1, lighting, road_surface, road_cond_1,
        #   type_of_collision, collision_severity, primary_rd,
        #   analysis_neighborhood
        "crash_select": (
            "collision_datetime,weather_1,lighting,road_surface,"
            "road_cond_1,type_of_collision,collision_severity,"
            "primary_rd,tb_latitude,tb_longitude,analysis_neighborhood"
        ),
        "crash_where_tpl": "collision_datetime >= '{start_date}'",
        "crash_col_map": {
            "CRASH_DATE":            "collision_datetime",
            "WEATHER_CONDITION":     "weather_1",
            "LIGHTING_CONDITION":    "lighting",
            "ROADWAY_SURFACE_COND":  "road_surface",
            "ROAD_DEFECT":           "road_cond_1",
            "ALIGNMENT":             None,
            "TRAFFICWAY_TYPE":       None,
            "LANE_CNT":              None,
            "POSTED_SPEED_LIMIT":    None,
            "TRAFFIC_CONTROL_DEVICE":None,
            "DEVICE_CONDITION":      None,
            "INTERSECTION_RELATED_I":None,
            "FIRST_CRASH_TYPE":      "type_of_collision",
            "CRASH_TYPE":            "collision_severity",
            "DAMAGE":                None,
            "NUM_UNITS":             None,
            "HIT_AND_RUN_I":         None,
            "LATITUDE":              "tb_latitude",
            "LONGITUDE":             "tb_longitude",
            "BEAT_OF_OCCURRENCE":    "analysis_neighborhood",
        },

        "geojson_url": "https://data.sfgov.org/api/geospatial/p5b7-5n3h?method=export&type=GeoJSON",
        "geojson_id_field": "nhood",
        "geojson_name_field": "nhood",
        "center": {"lat": 37.7749, "lon": -122.4194},
        "lat_bounds": (37.7, 37.85),
        "zoom": 11,

        "region_label": "Neighborhood",
        "region_names": {},  # populated dynamically from GeoJSON

        "ward_geojson_url": None,

        # Census API — census tracts in SF County (state 06, county 075)
        "census_state": "06",
        "census_county": "075",

        "census_csv": "census.csv",
        "census_area_col": "GEOID",
        "census_name_col": "NAME",
        "census_hardship_col": "HARDSHIP INDEX",
        "census_feature_cols": [
            "PERCENT OF HOUSING CROWDED",
            "PERCENT HOUSEHOLDS BELOW POVERTY",
            "PERCENT AGED 16+ UNEMPLOYED",
            "PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA",
            "PERCENT AGED UNDER 18 OR OVER 64",
            "PER CAPITA INCOME",
        ],
        "census_feature_short": [
            "Housing Crowded", "Below Poverty", "Unemployed 16+",
            "No HS Diploma", "Under 18/Over 64", "Per Capita Income",
        ],

        "transit_csvs": {},
    },

    # ════════════════════════════════════════════
    # AUSTIN
    # Verified: data.austintexas.gov
    # Crime:  fdj4-gpfu  (Crime Reports)
    # Crash:  y2wy-tgr5  (Austin Crash Report Data - Crash Level Records)
    # ════════════════════════════════════════════
    "austin": {
        "display_name": "Austin",
        "portal_url": "https://data.austintexas.gov/resource",
        "app_token_env": "AUSTIN_DATA_PORTAL_TOKEN",

        "crime_dataset_id": "fdj4-gpfu",
        "crash_dataset_id": "y2wy-tgr5",

        # Verified columns: occ_date_time, crime_type, council_district
        # NOTE: actual column is occ_date_time (NOT occurred_date_time)
        #       and crime_type (NOT highest_offense_description)
        "crime_select": "council_district,occ_date_time,crime_type",
        "crime_where_tpl": "occ_date_time >= '{start_year}-01-01T00:00:00'",
        "crime_col_map": {
            "date": "occ_date_time",
            "type": "crime_type",
            "area": "council_district",
        },

        # Verified columns: crash_timestamp, latitude, longitude,
        #   crash_speed_limit, units_involved, collsn_desc, crash_sev_id,
        #   crash_fatal_fl
        "crash_select": (
            "crash_timestamp,latitude,longitude,"
            "crash_speed_limit,units_involved,collsn_desc,crash_sev_id"
        ),
        "crash_where_tpl": "crash_timestamp >= '{start_date}'",
        "crash_col_map": {
            "CRASH_DATE":            "crash_timestamp",
            "WEATHER_CONDITION":     None,
            "LIGHTING_CONDITION":    None,
            "ROADWAY_SURFACE_COND":  None,
            "ROAD_DEFECT":           None,
            "ALIGNMENT":             None,
            "TRAFFICWAY_TYPE":       None,
            "LANE_CNT":              None,
            "POSTED_SPEED_LIMIT":    "crash_speed_limit",
            "TRAFFIC_CONTROL_DEVICE":None,
            "DEVICE_CONDITION":      None,
            "INTERSECTION_RELATED_I":None,
            "FIRST_CRASH_TYPE":      "collsn_desc",
            "CRASH_TYPE":            "crash_sev_id",
            "DAMAGE":                None,
            "NUM_UNITS":             "units_involved",
            "HIT_AND_RUN_I":         None,
            "LATITUDE":              "latitude",
            "LONGITUDE":             "longitude",
            "BEAT_OF_OCCURRENCE":    None,
        },

        "geojson_url": "https://data.austintexas.gov/api/geospatial/b54d-kih5?method=export&type=GeoJSON",
        "geojson_id_field": "district_number",
        "geojson_name_field": "district_number",
        "center": {"lat": 30.2672, "lon": -97.7431},
        "lat_bounds": (30.1, 30.5),
        "zoom": 10,

        "region_label": "Council District",
        "region_names": {str(i): f"District {i}" for i in range(1, 11)},

        "ward_geojson_url": None,

        # Census API — census tracts in Travis County (state 48, county 453)
        "census_state": "48",
        "census_county": "453",

        "census_csv": "census.csv",
        "census_area_col": "GEOID",
        "census_name_col": "NAME",
        "census_hardship_col": "HARDSHIP INDEX",
        "census_feature_cols": [
            "PERCENT OF HOUSING CROWDED",
            "PERCENT HOUSEHOLDS BELOW POVERTY",
            "PERCENT AGED 16+ UNEMPLOYED",
            "PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA",
            "PERCENT AGED UNDER 18 OR OVER 64",
            "PER CAPITA INCOME",
        ],
        "census_feature_short": [
            "Housing Crowded", "Below Poverty", "Unemployed 16+",
            "No HS Diploma", "Under 18/Over 64", "Per Capita Income",
        ],

        "transit_csvs": {},
    },

    # ════════════════════════════════════════════
    # SEATTLE
    # Verified: data.seattle.gov
    # Crime:  tazs-3rd5  (SPD Crime Data: 2008-Present)
    # Crash:  qdnv-25h8  (SDOT Collisions All Years)
    # ════════════════════════════════════════════
    "seattle": {
        "display_name": "Seattle",
        "portal_url": "https://data.seattle.gov/resource",
        "app_token_env": "SEATTLE_DATA_PORTAL_TOKEN",

        "crime_dataset_id": "tazs-3rd5",
        "crash_dataset_id": "qdnv-25h8",

        # Verified columns: offense_date, offense_category, neighborhood
        # NOTE: actual columns differ from initial config —
        #       offense_date (NOT offense_start_datetime),
        #       offense_category (NOT offense),
        #       neighborhood (NOT mcpp)
        "crime_select": "neighborhood,offense_date,offense_category",
        "crime_where_tpl": "offense_date >= '{start_year}-01-01T00:00:00'",
        "crime_col_map": {
            "date": "offense_date",
            "type": "offense_category",
            "area": "neighborhood",
        },

        # SDOT Collisions All Years (qdnv-25h8)
        # Standard SDOT collision columns (WSDOT schema):
        #   INCDTTM, WEATHER, LIGHTCOND, ROADCOND, JUNCTIONTYPE,
        #   COLLISIONTYPE, SEVERITYDESC, PERSONCOUNT, HITPARKEDCAR,
        #   WGS84LAT, WGS84LONG, SPEEDING
        "crash_select": (
            "incdttm,weather,lightcond,roadcond,junctiontype,"
            "collisiontype,severitydesc,personcount,"
            "hitparkedcar,wgs84lat,wgs84long,speeding"
        ),
        "crash_where_tpl": "incdttm >= '{start_date}'",
        "crash_col_map": {
            "CRASH_DATE":            "incdttm",
            "WEATHER_CONDITION":     "weather",
            "LIGHTING_CONDITION":    "lightcond",
            "ROADWAY_SURFACE_COND":  "roadcond",
            "ROAD_DEFECT":           None,
            "ALIGNMENT":             None,
            "TRAFFICWAY_TYPE":       "junctiontype",
            "LANE_CNT":              None,
            "POSTED_SPEED_LIMIT":    None,
            "TRAFFIC_CONTROL_DEVICE":None,
            "DEVICE_CONDITION":      None,
            "INTERSECTION_RELATED_I":None,
            "FIRST_CRASH_TYPE":      "collisiontype",
            "CRASH_TYPE":            "severitydesc",
            "DAMAGE":                "severitydesc",
            "NUM_UNITS":             "personcount",
            "HIT_AND_RUN_I":         "hitparkedcar",
            "LATITUDE":              "wgs84lat",
            "LONGITUDE":             "wgs84long",
            "BEAT_OF_OCCURRENCE":    None,
        },

        "geojson_url": "https://raw.githubusercontent.com/seattleio/seattle-boundaries-data/master/data/neighborhoods.geojson",
        "geojson_id_field": "name",
        "geojson_name_field": "name",
        "center": {"lat": 47.6062, "lon": -122.3321},
        "lat_bounds": (47.49, 47.74),
        "zoom": 10,

        "region_label": "Neighborhood",
        "region_names": {},  # populated dynamically from GeoJSON

        "ward_geojson_url": None,

        # Census API — census tracts in King County (state 53, county 033)
        "census_state": "53",
        "census_county": "033",

        "census_csv": "census.csv",
        "census_area_col": "GEOID",
        "census_name_col": "NAME",
        "census_hardship_col": "HARDSHIP INDEX",
        "census_feature_cols": [
            "PERCENT OF HOUSING CROWDED",
            "PERCENT HOUSEHOLDS BELOW POVERTY",
            "PERCENT AGED 16+ UNEMPLOYED",
            "PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA",
            "PERCENT AGED UNDER 18 OR OVER 64",
            "PER CAPITA INCOME",
        ],
        "census_feature_short": [
            "Housing Crowded", "Below Poverty", "Unemployed 16+",
            "No HS Diploma", "Under 18/Over 64", "Per Capita Income",
        ],

        "transit_csvs": {},
    },
}


def get_city(city_key: str) -> dict:
    """Return the configuration dict for *city_key*, enriched with computed paths."""
    cfg = CITIES[city_key].copy()
    d = _data_dir(city_key)
    cfg["data_dir"] = d
    cfg["crime_csv"] = os.path.join(d, "crime_monthly_pivot.csv")
    cfg["crash_csv"] = os.path.join(d, "traffic_crashes_latest.csv")
    cfg["census_csv_path"] = os.path.join(d, cfg["census_csv"])
    cfg["city_key"] = city_key

    # Socioeconomic choropleth GeoJSON.
    # Chicago uses its community-area GeoJSON (web URL, matches the pre-loaded CSV).
    # Other cities use locally-cached TIGER/Line census-tract boundaries so that
    # the GeoJSON IDs (11-char GEOIDs) match the ACS data fetched by fetch_census().
    if city_key == "chicago":
        cfg["census_geojson_url"]        = cfg["geojson_url"]
        cfg["census_geojson_id_field"]   = cfg["geojson_id_field"]
        cfg["census_geojson_name_field"] = cfg["geojson_name_field"]
    else:
        cfg["census_geojson_url"]        = os.path.join(d, "census_tracts.geojson")
        cfg["census_geojson_id_field"]   = "GEOID"
        cfg["census_geojson_name_field"] = "NAMELSAD"

    return cfg


def city_keys() -> list[str]:
    """Return all configured city keys in display order."""
    return list(CITIES.keys())


def city_display_names() -> dict[str, str]:
    """Return {key: display_name} for the UI dropdown."""
    return {k: v["display_name"] for k, v in CITIES.items()}
=======
Per-city registry. One source of truth for paths, endpoints, geometry, schema.
All other modules accept a `CityConfig` instead of hardcoding Chicago.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import requests


SRC_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class CityConfig:
    key: str
    name: str
    data_dir: str

    # Per-city CSV filenames (resolved under SRC_DIR/data_dir/)
    crime_csv: str = "crime_monthly_pivot.csv"
    crash_csv: str = "traffic_crashes_latest.csv"
    census_csv: str = "census.csv"

    # Map view
    center: Tuple[float, float] = (0.0, 0.0)
    zoom: int = 10
    lat_bounds: Tuple[float, float] = (-90.0, 90.0)
    lon_bounds: Tuple[float, float] = (-180.0, 180.0)

    # Boundary geometry (Socrata GeoJSON URL OR local file path)
    boundary_url: Optional[str] = None
    boundary_path: Optional[str] = None        # relative to data_dir
    boundary_id_field: str = "GEOID"
    boundary_name_field: str = "NAME"

    # Schema mapping
    crime_area_col: str = "Community Area"
    census_id_col: str = "GEOID"

    # Area-ID semantics: how to normalize CSV `crime_area_col` values so they
    # match boundary `boundary_id_field` keys. "int" (Chicago community areas,
    # NYC precincts), "str" (SF neighborhoods), "upper_str" (LAPD divisions).
    area_id_kind: str = "int"
    # CSV value → boundary ID overrides applied before area_id_kind coercion.
    crime_area_aliases: Tuple[Tuple[str, str], ...] = ()
    # Prefix prepended to area names in UI labels (e.g. "Precinct ").
    area_display_prefix: str = ""

    # Data Portal (for refresh)
    soda_portal: Optional[str] = None
    crime_dataset_id: Optional[str] = None
    crash_dataset_id: Optional[str] = None
    token_env: str = ""

    # Capability flags
    has_transport_layer: bool = False

    # Helpers --------------------------------------------------------------
    def path(self, filename: str) -> str:
        return os.path.join(SRC_DIR, self.data_dir, filename)

    @property
    def crime_path(self) -> str:
        return self.path(self.crime_csv)

    @property
    def crash_path(self) -> str:
        return self.path(self.crash_csv)

    @property
    def census_path(self) -> str:
        return self.path(self.census_csv)

    def normalize_area_key(self, value):
        """Normalize a CSV area value to match the boundary id space."""
        import pandas as _pd
        if value is None or (isinstance(value, float) and _pd.isna(value)):
            return None
        alias = dict(self.crime_area_aliases)
        v = alias.get(value, value)
        if isinstance(v, str):
            v = alias.get(v, v)
        kind = self.area_id_kind
        if kind == "int":
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return None
        if kind == "upper_str":
            return str(v).strip().upper()
        return str(v).strip()


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

CITIES: dict[str, CityConfig] = {
    "chicago": CityConfig(
        key="chicago",
        name="Chicago",
        data_dir="chicago",
        crime_csv="crime_monthly_pivot.csv",
        crash_csv="traffic_crashes_latest.csv",
        census_csv="censusChicago.csv",
        center=(41.8781, -87.6298),
        zoom=10,
        lat_bounds=(41.6, 42.1),
        lon_bounds=(-87.95, -87.5),
        boundary_url="https://data.cityofchicago.org/resource/igwz-8jzy.geojson",
        boundary_id_field="area_num_1",
        boundary_name_field="community",
        crime_area_col="Community Area",
        census_id_col="Community Area Number",
        soda_portal="https://data.cityofchicago.org/resource",
        crime_dataset_id="ijzp-q8t2",
        crash_dataset_id="85ca-t3if",
        token_env="CHICAGO_DATA_PORTAL_TOKEN",
        has_transport_layer=True,
    ),
    "new_york": CityConfig(
        key="new_york",
        name="New York",
        data_dir="new_york",
        center=(40.7128, -74.0060),
        zoom=10,
        lat_bounds=(40.4, 41.0),
        lon_bounds=(-74.3, -73.6),
        boundary_url="https://data.cityofnewyork.us/resource/y76i-bdw7.geojson?$limit=200",
        boundary_id_field="precinct",
        boundary_name_field="precinct",
        crime_area_col="Community Area",
        census_id_col="GEOID",
        area_id_kind="int",
        area_display_prefix="Precinct ",
        soda_portal="https://data.cityofnewyork.us/resource",
        crime_dataset_id="qgea-i56i",      # NYPD Complaint Data Historic
        crash_dataset_id="h9gi-nx95",      # Motor Vehicle Collisions - Crashes
        token_env="NYC_DATA_PORTAL_TOKEN",
    ),
    "los_angeles": CityConfig(
        key="los_angeles",
        name="Los Angeles",
        data_dir="los_angeles",
        center=(34.0522, -118.2437),
        zoom=10,
        lat_bounds=(33.7, 34.35),
        lon_bounds=(-118.7, -118.1),
        boundary_path="lapd_divisions.geojson",
        boundary_id_field="APREC",
        boundary_name_field="APREC",
        crime_area_col="Community Area",
        census_id_col="GEOID",
        area_id_kind="upper_str",
        crime_area_aliases=(
            ("N Hollywood", "NORTH HOLLYWOOD"),
            ("West LA", "WEST LOS ANGELES"),
        ),
        soda_portal="https://data.lacity.org/resource",
        crime_dataset_id="2nrs-mtv8",
        crash_dataset_id="d5tf-ez2w",
        token_env="LA_DATA_PORTAL_TOKEN",
    ),
    "san_francisco": CityConfig(
        key="san_francisco",
        name="San Francisco",
        data_dir="san_francisco",
        center=(37.7749, -122.4194),
        zoom=11,
        lat_bounds=(37.6, 37.85),
        lon_bounds=(-122.55, -122.35),
        boundary_url="https://data.sfgov.org/resource/ajp5-b2md.geojson",
        boundary_id_field="nhood",
        boundary_name_field="nhood",
        crime_area_col="Community Area",
        census_id_col="GEOID",
        area_id_kind="str",
        crime_area_aliases=(
            ("Financial District/South Beach", "Financial District"),
        ),
        soda_portal="https://data.sfgov.org/resource",
        crime_dataset_id="wg3w-h783",
        crash_dataset_id="ubvf-ztfx",
        token_env="SF_DATA_PORTAL_TOKEN",
    ),
    "philadelphia": CityConfig(
        key="philadelphia",
        name="Philadelphia",
        data_dir="philadelphia",
        center=(39.9526, -75.1652),
        zoom=11,
        lat_bounds=(39.85, 40.15),
        lon_bounds=(-75.3, -74.95),
        boundary_path="police_districts.geojson",
        boundary_id_field="dist_num",
        boundary_name_field="dist_num",
        crime_area_col="Community Area",
        census_id_col="GEOID",
        area_id_kind="int",
        area_display_prefix="District ",
        soda_portal="https://phl.carto.com/api/v2/sql",
        token_env="PHL_DATA_PORTAL_TOKEN",
    ),
}

DEFAULT_CITY_KEY = "chicago"


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def get_city(key: str) -> CityConfig:
    return CITIES[key]


def list_cities() -> list[tuple[str, str]]:
    return [(k, c.name) for k, c in CITIES.items()]


def load_boundary(city: CityConfig) -> tuple[dict, dict]:
    """
    Return (geojson_dict, area_map). Tries local boundary_path first
    (under data_dir), then boundary_url. area_map is {id: name}.
    """
    geo: Optional[dict] = None

    # Try local file first — but skip empty placeholder files
    if city.boundary_path:
        local = city.path(city.boundary_path)
        if os.path.exists(local):
            with open(local, "r") as f:
                candidate = json.load(f)
            if candidate.get("features"):
                geo = candidate

    # Fallback to remote URL
    if geo is None and city.boundary_url:
        try:
            resp = requests.get(city.boundary_url, timeout=60)
            resp.raise_for_status()
            geo = resp.json()
        except Exception:
            geo = None

    if geo is None:
        raise FileNotFoundError(
            f"No boundary geometry available for {city.name} "
            f"(boundary_path={city.boundary_path}, boundary_url={city.boundary_url})"
        )

    id_field = city.boundary_id_field
    name_field = city.boundary_name_field

    area_map: dict = {}
    for feat in geo.get("features", []):
        props = feat.get("properties", {}) or {}
        raw_id = props.get(id_field)
        if raw_id is None:
            continue
        if city.area_id_kind == "int":
            try:
                key_id = int(float(raw_id))
            except (TypeError, ValueError):
                continue
        elif city.area_id_kind == "upper_str":
            key_id = str(raw_id).strip().upper()
        else:
            key_id = str(raw_id).strip()
        raw_name = props.get(name_field, str(raw_id))
        area_map[key_id] = (
            f"{city.area_display_prefix}{raw_name}" if city.area_display_prefix
            else raw_name
        )

    return geo, area_map
>>>>>>> Stashed changes
