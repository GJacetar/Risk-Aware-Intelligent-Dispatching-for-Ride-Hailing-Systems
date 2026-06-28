import argparse
import json
import math
import os
import re
import warnings
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import requests
from joblib import Parallel
from joblib import delayed
from scipy.spatial import cKDTree
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

try:
    import osmnx as ox
except Exception:
    ox = None

CITY_NAME = "NYC"
ZONE_MODE = "nyc"
BASE_DIR = Path("/root/autodl-tmp/new/data")
RAW_DIR = Path("/root/autodl-tmp/new/data")
OUTPUT_DIR = Path("/root/autodl-tmp/new/data/processed")
GRAPH_PATH = Path("/root/autodl-tmp/new/data/nyc_drive_network.graphml")
POI_PATH = Path("/root/autodl-tmp/new/data/nyc_pois.geojson")
START_TIME = "2026-01-01 00:00:00"
END_TIME = "2026-03-01 00:00:00"
SLOT_MINUTES = 5
CENTER_LAT = 40.7128
CENTER_LON = -74.0060
TIMEZONE = "America/New_York"
RADIUS_KM = 0.0
SIGMA_KM = 3.0
DEFAULT_MAX_DISTANCE_KM = 30.0
ZONE_CANDIDATES = ['/root/autodl-tmp/new/data/taxi_zones.shp', '/root/autodl-tmp/new/data/taxi_zones/taxi_zones.shp']
TRIP_FILES = ['/root/autodl-tmp/new/data/fhvhv_tripdata_2026-01.parquet', '/root/autodl-tmp/new/data/fhvhv_tripdata_2026-02.parquet']
POI_TAGS = ["amenity", "building", "public_transport", "leisure", "shop", "office"]

def log(message):
    print(message, flush=True)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def norm_col(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")

def clean_numeric_id(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    if ZONE_MODE == "nyc":
        m = re.search(r"\d+", s)
        return str(int(m.group(0))) if m else None
    if ZONE_MODE == "chicago_community":
        m = re.search(r"\d+", s)
        return str(int(m.group(0))) if m else None
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"[^\d]", "", s)
    return s if s else None

def find_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None

def find_col(columns, candidates, required=True):
    lookup = {norm_col(c): c for c in columns}
    for cand in candidates:
        key = norm_col(cand)
        if key in lookup:
            return lookup[key]
    if required:
        raise ValueError(f"Missing column from candidates {candidates}")
    return None

def money_to_float(series):
    return pd.to_numeric(series.astype(str).str.replace(r"[^\d.\-]", "", regex=True).replace({"": np.nan}), errors="coerce")

def read_zones():
    zone_path = find_existing(ZONE_CANDIDATES)
    if zone_path is None:
        found = list(RAW_DIR.rglob("*.shp")) + list(RAW_DIR.rglob("*.geojson")) + list(RAW_DIR.rglob("*.json"))
        if found:
            zone_path = found[0]
    if zone_path is None:
        raise FileNotFoundError("No zone boundary file found")
    zones = gpd.read_file(zone_path).to_crs("EPSG:4326")
    zones = zones[zones.geometry.notna() & ~zones.geometry.is_empty].copy()
    columns = list(zones.columns)
    if ZONE_MODE == "nyc":
        id_col = find_col(columns, ["LocationID", "location_id", "zone_id", "OBJECTID"])
        name_col = find_col(columns, ["zone", "Zone", "borough", "Borough"], required=False) or id_col
        zones["zone_raw_id"] = zones[id_col].apply(clean_numeric_id)
        zones["zone_name"] = zones[name_col].astype(str)
    elif ZONE_MODE == "chicago_tract":
        id_col = find_col(columns, ["geoid10", "geoid", "GEOID10", "tractce10", "tract"], required=False) or columns[0]
        name_col = id_col
        zones["zone_raw_id"] = zones[id_col].apply(clean_numeric_id)
        zones["zone_name"] = zones[id_col].astype(str)
        if RADIUS_KM > 0:
            zones_proj = zones.to_crs("EPSG:32616")
            center = gpd.points_from_xy([CENTER_LON], [CENTER_LAT], crs="EPSG:4326").to_crs("EPSG:32616")[0]
            keep = zones_proj.geometry.centroid.distance(center) <= RADIUS_KM * 1000.0
            zones = zones.loc[keep.values].copy()
    else:
        id_col = find_col(columns, ["area_numbe", "area_num_1", "area_num", "community_area", "community_area_number", "comarea", "area"], required=False)
        if id_col is None:
            best = None
            best_count = 0
            for c in columns:
                if c == zones.geometry.name:
                    continue
                vals = zones[c].apply(clean_numeric_id).dropna()
                count = len(set(vals))
                if count > best_count:
                    best = c
                    best_count = count
            id_col = best
        name_col = find_col(columns, ["community", "community_area_name", "community_name", "name", "area_name"], required=False) or id_col
        zones["zone_raw_id"] = zones[id_col].apply(clean_numeric_id)
        zones["zone_name"] = zones[name_col].astype(str)
    zones = zones.dropna(subset=["zone_raw_id"]).copy()
    if zones["zone_raw_id"].duplicated().any():
        zones = zones.dissolve(by="zone_raw_id", as_index=False, aggfunc="first")
    if ZONE_MODE in {"nyc", "chicago_community"}:
        zones = zones.sort_values("zone_raw_id", key=lambda s: s.astype(int)).reset_index(drop=True)
    else:
        zones = zones.sort_values("zone_raw_id").reset_index(drop=True)
    zones["zone_idx"] = np.arange(len(zones), dtype=int)
    return zones[["zone_idx", "zone_raw_id", "zone_name", "geometry"]].copy()

def haversine(lon1, lat1, lon2, lat2):
    r = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def centroid_distance_matrix(zones):
    reps = zones.geometry.representative_point()
    lon = reps.x.values
    lat = reps.y.values
    n = len(zones)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        dist[i, :] = haversine(lon[i], lat[i], lon, lat)
    return np.clip(dist, 0.0, DEFAULT_MAX_DISTANCE_KM).astype(np.float32)

def nearest_nodes_manual(graph, lons, lats):
    nodes = []
    coords = []
    for node, data in graph.nodes(data=True):
        if "x" in data and "y" in data:
            nodes.append(node)
            coords.append((float(data["x"]), float(data["y"])))
    if not coords:
        raise RuntimeError("The graph has no node coordinates")
    tree = cKDTree(np.asarray(coords, dtype=float))
    _, idx = tree.query(np.c_[lons, lats], k=1)
    return [nodes[int(i)] for i in idx]

def read_graph():
    if not GRAPH_PATH.exists():
        return None
    if ox is not None:
        try:
            graph = ox.load_graphml(str(GRAPH_PATH))
        except Exception:
            graph = nx.read_graphml(str(GRAPH_PATH))
    else:
        graph = nx.read_graphml(str(GRAPH_PATH))
    for _, data in graph.nodes(data=True):
        if "x" in data:
            data["x"] = float(data["x"])
        if "y" in data:
            data["y"] = float(data["y"])
    if isinstance(graph, (nx.MultiDiGraph, nx.MultiGraph)):
        for _, _, _, data in graph.edges(keys=True, data=True):
            data["length"] = float(data.get("length", 1.0))
    else:
        for _, _, data in graph.edges(data=True):
            data["length"] = float(data.get("length", 1.0))
    return graph

def graph_distance_matrix(zones):
    graph = read_graph()
    if graph is None:
        return centroid_distance_matrix(zones)
    reps = zones.geometry.representative_point()
    lon = reps.x.values
    lat = reps.y.values
    try:
        if ox is None:
            raise RuntimeError("osmnx unavailable")
        nearest = ox.distance.nearest_nodes(graph, lon, lat)
    except Exception:
        nearest = nearest_nodes_manual(graph, lon, lat)
    n = len(zones)
    def calc(i):
        try:
            lengths = nx.single_source_dijkstra_path_length(graph, nearest[i], weight="length")
        except Exception:
            lengths = {}
        row = np.full(n, DEFAULT_MAX_DISTANCE_KM, dtype=np.float32)
        row[i] = 0.0
        for j in range(n):
            if j != i and nearest[j] in lengths:
                row[j] = float(lengths[nearest[j]]) / 1000.0
        return i, row
    results = Parallel(n_jobs=-1, backend="loky")(delayed(calc)(i) for i in range(n))
    dist = np.full((n, n), DEFAULT_MAX_DISTANCE_KM, dtype=np.float32)
    for i, row in results:
        dist[i, :] = row
    dist = np.minimum(dist, dist.T)
    return np.clip(dist, 0.0, DEFAULT_MAX_DISTANCE_KM).astype(np.float32)

def build_graphs(zones):
    ensure_dir(OUTPUT_DIR)
    dist = graph_distance_matrix(zones)
    adj_spatial = np.exp(-(dist ** 2) / (SIGMA_KM ** 2)).astype(np.float32)
    np.fill_diagonal(adj_spatial, 1.0)
    np.save(OUTPUT_DIR / "dist_matrix.npy", dist)
    np.save(OUTPUT_DIR / "adj_spatial.npy", adj_spatial)
    zones.to_file(OUTPUT_DIR / "zones_processed.geojson", driver="GeoJSON")
    zones[["zone_idx", "zone_raw_id", "zone_name"]].to_csv(OUTPUT_DIR / "zone_mapping.csv", index=False)
    build_semantic_graph(zones)

def is_valid_poi(value):
    if pd.isna(value):
        return False
    s = str(value).strip().lower()
    return s not in {"", "nan", "none", "false", "0"}

def build_semantic_graph(zones):
    n = len(zones)
    if not POI_PATH.exists():
        adj = np.eye(n, dtype=np.float32)
        np.save(OUTPUT_DIR / "adj_semantic.npy", adj)
        return
    pois = gpd.read_file(POI_PATH).to_crs("EPSG:4326")
    pois = pois[pois.geometry.notna() & ~pois.geometry.is_empty].copy()
    pois["geometry"] = pois.geometry.apply(lambda g: g if g.geom_type == "Point" else g.representative_point())
    joined = gpd.sjoin(pois, zones[["zone_idx", "geometry"]], how="inner", predicate="within")
    features = np.zeros((n, len(POI_TAGS)), dtype=np.float32)
    for _, row in joined.iterrows():
        z = int(row["zone_idx"])
        for j, tag in enumerate(POI_TAGS):
            if tag in joined.columns and is_valid_poi(row.get(tag)):
                features[z, j] += 1.0
    zero = features.sum(axis=1) <= 0
    if np.any(zero):
        features[zero, :] = 1e-6
    tfidf = TfidfTransformer().fit_transform(features).toarray()
    adj = cosine_similarity(tfidf).astype(np.float32)
    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    adj = np.clip(adj, 0.0, 1.0)
    np.fill_diagonal(adj, 1.0)
    np.save(OUTPUT_DIR / "adj_semantic.npy", adj)
    df = pd.DataFrame(features, columns=POI_TAGS)
    df.insert(0, "zone_idx", zones["zone_idx"].values)
    df.insert(1, "zone_raw_id", zones["zone_raw_id"].astype(str).values)
    df.insert(2, "zone_name", zones["zone_name"].astype(str).values)
    df.to_csv(OUTPUT_DIR / "poi_feature_matrix.csv", index=False)

def read_one_trip_file(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)

def load_trips():
    frames = []
    for file_path in TRIP_FILES:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        df = read_one_trip_file(p)
        df.columns = [norm_col(c) for c in df.columns]
        df["_source_file"] = p.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def choose_trip_columns(df):
    cols = list(df.columns)
    start_col = find_col(cols, ["pickup_datetime", "trip_start_timestamp", "pickup_time", "start_time"])
    end_col = find_col(cols, ["dropoff_datetime", "trip_end_timestamp", "dropoff_time", "end_time"], required=False)
    if ZONE_MODE == "nyc":
        pu_col = find_col(cols, ["pulocationid", "pu_location_id", "pickup_location_id"])
        do_col = find_col(cols, ["dolocationid", "do_location_id", "dropoff_location_id"])
    elif ZONE_MODE == "chicago_tract":
        pu_col = find_col(cols, ["pickup_census_tract", "pickup_tract", "pu_census_tract"])
        do_col = find_col(cols, ["dropoff_census_tract", "dropoff_tract", "do_census_tract"])
    else:
        pu_col = find_col(cols, ["pickup_community_area", "pickup_area", "pu_community_area"])
        do_col = find_col(cols, ["dropoff_community_area", "dropoff_area", "do_community_area"])
    miles_col = find_col(cols, ["trip_miles", "miles", "trip_distance"], required=False)
    seconds_col = find_col(cols, ["trip_seconds", "duration_sec", "duration_seconds"], required=False)
    fare_col = find_col(cols, ["base_passenger_fare", "trip_total", "fare"], required=False)
    return start_col, end_col, pu_col, do_col, miles_col, seconds_col, fare_col

def build_trip_tensors(zones):
    zone_map = {str(a): int(b) for a, b in zip(zones["zone_raw_id"], zones["zone_idx"])}
    df = load_trips()
    start_col, end_col, pu_col, do_col, miles_col, seconds_col, fare_col = choose_trip_columns(df)
    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    if end_col is not None:
        df[end_col] = pd.to_datetime(df[end_col], errors="coerce")
    else:
        df[end_col] = pd.NaT
    df["pu_raw"] = df[pu_col].apply(clean_numeric_id)
    df["do_raw"] = df[do_col].apply(clean_numeric_id)
    df["pu_idx"] = df["pu_raw"].map(zone_map)
    df["do_idx"] = df["do_raw"].map(zone_map)
    df = df.dropna(subset=[start_col, "pu_idx", "do_idx"]).copy()
    df["pu_idx"] = df["pu_idx"].astype(int)
    df["do_idx"] = df["do_idx"].astype(int)
    start_time = pd.to_datetime(START_TIME)
    end_time = pd.to_datetime(END_TIME)
    max_slots = int((end_time - start_time).total_seconds() // (SLOT_MINUTES * 60))
    df["pickup_slot"] = np.floor((df[start_col] - start_time).dt.total_seconds() / (SLOT_MINUTES * 60))
    df = df[(df["pickup_slot"] >= 0) & (df["pickup_slot"] < max_slots)].copy()
    df["pickup_slot"] = df["pickup_slot"].astype(int)
    if df[end_col].notna().any():
        df["dropoff_slot"] = np.floor((df[end_col] - start_time).dt.total_seconds() / (SLOT_MINUTES * 60))
    else:
        df["dropoff_slot"] = np.nan
    if seconds_col is not None:
        df["trip_seconds"] = pd.to_numeric(df[seconds_col], errors="coerce")
    else:
        df["trip_seconds"] = (df[end_col] - df[start_col]).dt.total_seconds()
    if miles_col is not None:
        df["trip_miles"] = pd.to_numeric(df[miles_col], errors="coerce")
    else:
        df["trip_miles"] = np.nan
    if fare_col is not None:
        df["base_passenger_fare"] = money_to_float(df[fare_col])
    else:
        df["base_passenger_fare"] = np.nan
    df = df[(df["trip_seconds"] >= 60) & (df["trip_seconds"] <= 7200)].copy()
    df = df[(df["trip_miles"].fillna(1.0) > 0) & (df["trip_miles"].fillna(1.0) <= 100)].copy()
    df["avg_speed"] = (df["trip_miles"] / df["trip_seconds"]) * 3600.0
    df.loc[(df["avg_speed"] < 1.0) | (df["avg_speed"] > 80.0), "avg_speed"] = np.nan
    n = len(zones)
    demand = np.zeros((max_slots, n, 1), dtype=np.float32)
    speed = np.zeros((max_slots, n, 1), dtype=np.float32)
    enroute = np.zeros((max_slots, n, 1), dtype=np.float32)
    counts = df.groupby(["pickup_slot", "pu_idx"]).size().reset_index(name="count")
    demand[counts["pickup_slot"].values.astype(int), counts["pu_idx"].values.astype(int), 0] = counts["count"].values.astype(np.float32)
    speed_df = df.groupby(["pickup_slot", "pu_idx"])["avg_speed"].mean().reset_index().dropna(subset=["avg_speed"])
    speed[speed_df["pickup_slot"].values.astype(int), speed_df["pu_idx"].values.astype(int), 0] = speed_df["avg_speed"].values.astype(np.float32)
    er = df.dropna(subset=["dropoff_slot"]).copy()
    er = er[er["dropoff_slot"] > er["pickup_slot"]]
    er = er[er["dropoff_slot"] < max_slots]
    er_counts = er.groupby(["dropoff_slot", "do_idx"]).size().reset_index(name="count")
    enroute[er_counts["dropoff_slot"].values.astype(int), er_counts["do_idx"].values.astype(int), 0] = er_counts["count"].values.astype(np.float32)
    np.save(OUTPUT_DIR / "demand_tensor_full.npy", demand)
    np.save(OUTPUT_DIR / "speed_tensor_full.npy", speed)
    np.save(OUTPUT_DIR / "enroute_tensor_full.npy", enroute)
    clean_cols = [start_col, end_col, "pu_raw", "do_raw", "pu_idx", "do_idx", "trip_miles", "trip_seconds", "avg_speed", "base_passenger_fare", "pickup_slot", "dropoff_slot", "_source_file"]
    clean_cols = [c for c in clean_cols if c in df.columns]
    df[clean_cols].to_csv(OUTPUT_DIR / f"{CITY_NAME}_clean_orders.csv", index=False)

def build_weather(zones):
    start = pd.to_datetime(START_TIME)
    end = pd.to_datetime(END_TIME)
    total_slots = int((end - start).total_seconds() // (SLOT_MINUTES * 60))
    target_index = pd.date_range(start=start, periods=total_slots, freq=f"{SLOT_MINUTES}min")
    params = {
        "latitude": CENTER_LAT,
        "longitude": CENTER_LON,
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "hourly": "temperature_2m,precipitation",
        "timezone": TIMEZONE
    }
    response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(response.text[:1000])
    data = response.json()["hourly"]
    weather = pd.DataFrame(data)
    weather["time"] = pd.to_datetime(weather["time"]).dt.tz_localize(None)
    weather = weather.set_index("time").sort_index()
    weather = weather[["temperature_2m", "precipitation"]].reindex(target_index, method="ffill").bfill().fillna(0.0)
    arr = weather.values.astype(np.float32)
    arr = np.expand_dims(arr, axis=1).repeat(len(zones), axis=1)
    np.save(OUTPUT_DIR / "weather_tensor.npy", arr)
    (OUTPUT_DIR / "weather_features.json").write_text(json.dumps(["temperature_2m", "precipitation"]), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_weather", action="store_true")
    args = parser.parse_args()
    ensure_dir(OUTPUT_DIR)
    zones = read_zones()
    log(f"{CITY_NAME} zones={len(zones)}")
    build_graphs(zones)
    build_trip_tensors(zones)
    if not args.skip_weather:
        build_weather(zones)
    log("done")

if __name__ == "__main__":
    main()
