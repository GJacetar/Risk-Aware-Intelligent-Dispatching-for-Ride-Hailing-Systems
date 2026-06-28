import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from model import CC_STMT

SIM_SLOT_MINUTES = 5
SEQ_LEN = 12
MAX_WAIT_SLOTS = 2
ADVANCE_DISPATCH_HORIZON_SLOTS = 2
ADVANCE_SERVICE_TOLERANCE_SLOTS = 2
AVG_EMPTY_SPEED_KMH = 28.0
UOT_GAMMA = 0.10

WEIGHT_PROFILES = {
    "default": {"dist": 1.05, "rev": 0.78, "wait": 0.95, "dest_val": 0.78, "dest_risk": 0.92},
    "paper": {"dist": 2.35, "rev": 0.88, "wait": 1.65, "dest_val": 0.30, "dest_risk": 0.56},
    "risk": {"dist": 1.80, "rev": 0.72, "wait": 1.20, "dest_val": 0.52, "dest_risk": 0.72}
}

@dataclass
class CityConfig:
    city: str
    root_dir: str
    proc_dir: str
    raw_dir: str
    out_dir: str
    model_path: str
    base_time: str
    pred_slot_minutes: int
    city_type: str
    default_fleets: Tuple[int, ...]
    default_max_orders: int
    default_max_pickup_dist: float

@dataclass
class OrderState:
    id: int
    pickup_zone: int
    dropoff_zone: int
    start_slot: int
    duration_slots: int
    revenue: float
    miles: float
    creation_slot: int
    is_advance: bool
    latest_pickup_slot: int
    advance_lead_slots: int

@dataclass
class VehicleState:
    id: int
    current_zone: int
    available_time: int = 0
    total_revenue: float = 0.0
    total_empty_miles: float = 0.0
    total_orders: int = 0

    def add_order(self, order: OrderState, empty_dist_km: float, current_slot: int) -> Tuple[int, int]:
        self.total_empty_miles += float(empty_dist_km)
        self.total_revenue += float(order.revenue)
        self.total_orders += 1
        empty_slots = int(math.ceil((float(empty_dist_km) / AVG_EMPTY_SPEED_KMH) * 60.0 / SIM_SLOT_MINUTES))
        service_start = max(int(current_slot + empty_slots), int(order.start_slot))
        self.available_time = int(service_start + order.duration_slots)
        self.current_zone = int(order.dropoff_zone)
        return service_start, self.available_time

@dataclass
class Variant:
    name: str
    use_value: bool
    use_risk: bool
    adjusted_value: bool
    alpha: float
    beta: float

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

def norm_col(c):
    return str(c).strip().lower().replace(" ", "_")

def find_col(cols, names, required=False):
    lookup = {norm_col(c): c for c in cols}
    for name in names:
        key = norm_col(name)
        if key in lookup:
            return lookup[key]
    if required:
        raise ValueError(f"missing column from {names}")
    return None

def clean_money(x):
    return pd.to_numeric(x.astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")

def clean_area_id(v):
    if pd.isna(v):
        return None
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    import re
    m = re.search(r"\d+", s)
    if m is None:
        return None
    return str(int(m.group(0)))

def safe_to_1d(x, n=None):
    arr = np.asarray(x)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        arr = np.array([float(arr)])
    elif arr.ndim == 2:
        if n is not None and arr.shape[0] == n:
            arr = arr.mean(axis=1)
        elif n is not None and arr.shape[1] == n:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=-1)
    elif arr.ndim > 2:
        if n is not None and n in arr.shape:
            axis = list(arr.shape).index(n)
            arr = arr.mean(axis=tuple(i for i in range(arr.ndim) if i != axis))
        else:
            arr = arr.reshape(-1)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if n is not None:
        if arr.size < n:
            arr = np.pad(arr, (0, n - arr.size), constant_values=float(arr[-1]) if arr.size else 0.0)
        arr = arr[:n]
    return arr

def norm_by_scale(x, scale):
    x = np.asarray(x, dtype=np.float32)
    if scale is None or not np.isfinite(scale) or scale <= 1e-8:
        scale = np.nanpercentile(np.abs(x), 95) + 1e-8
    return np.clip(x / (scale + 1e-8), 0.0, 1.0).astype(np.float32)

def minmax_masked(x, mask):
    out = np.zeros_like(x, dtype=np.float32)
    vals = x[mask]
    if vals.size == 0:
        return out
    lo = np.nanpercentile(vals, 5)
    hi = np.nanpercentile(vals, 95)
    if abs(hi - lo) < 1e-8:
        return out
    out[mask] = np.clip((x[mask] - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return out

def ensure_weather_shape(x, n):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        return np.expand_dims(x, 1).repeat(n, axis=1)
    if x.ndim == 3 and x.shape[1] == 1:
        return np.repeat(x, n, axis=1)
    if x.ndim == 3 and x.shape[1] == n:
        return x
    if x.ndim > 3:
        return ensure_weather_shape(np.squeeze(x), n)
    raise ValueError(f"bad weather shape {x.shape}")

def order_expired(o, current_slot):
    return int(current_slot) > int(o.latest_pickup_slot)

def order_dispatchable(o, current_slot):
    t = int(current_slot)
    if o.is_advance:
        return int(o.start_slot) - ADVANCE_DISPATCH_HORIZON_SLOTS <= t <= int(o.latest_pickup_slot)
    return int(o.start_slot) <= t <= int(o.latest_pickup_slot)

def wait_minutes(o, current_slot):
    if o.is_advance:
        return float(max(0, int(current_slot) - int(o.start_slot)) * SIM_SLOT_MINUTES)
    return float(max(0, int(current_slot) - int(o.creation_slot)) * SIM_SLOT_MINUTES)

def realized_wait_minutes(o, service_start):
    if o.is_advance:
        return float(max(0, int(service_start) - int(o.start_slot)) * SIM_SLOT_MINUTES)
    return float(max(0, int(service_start) - int(o.creation_slot)) * SIM_SLOT_MINUTES)

class Predictor:
    def __init__(self, cfg: CityConfig, device):
        self.cfg = cfg
        self.device = device
        p = Path(cfg.proc_dir)
        self.x_order = np.load(p / "demand_tensor_full.npy").astype(np.float32)
        self.x_speed = np.load(p / "speed_tensor_full.npy").astype(np.float32)
        self.x_enroute = np.load(p / "enroute_tensor_full.npy").astype(np.float32)
        weather = np.load(p / "weather_tensor.npy").astype(np.float32)
        self.num_nodes = int(self.x_order.shape[1])
        self.x_weather = ensure_weather_shape(weather, self.num_nodes)
        self.adj_spatial = torch.FloatTensor(np.load(p / "adj_spatial.npy").astype(np.float32)).to(device)
        self.adj_semantic = torch.FloatTensor(np.load(p / "adj_semantic.npy").astype(np.float32)).to(device)
        self.train_end = int(len(self.x_order) * 0.7)
        self.val_end = int(len(self.x_order) * 0.8)
        self.m_order = float(self.x_order[:self.train_end].mean())
        self.s_order = float(self.x_order[:self.train_end].std() + 1e-8)
        self.m_speed = float(self.x_speed[:self.train_end].mean())
        self.s_speed = float(self.x_speed[:self.train_end].std() + 1e-8)
        self.m_enroute = float(self.x_enroute[:self.train_end].mean())
        self.s_enroute = float(self.x_enroute[:self.train_end].std() + 1e-8)
        self.m_w = self.x_weather[:self.train_end].mean(axis=(0, 1))
        self.s_w = self.x_weather[:self.train_end].std(axis=(0, 1)) + 1e-8
        self.pred_step_ratio = max(1, int(cfg.pred_slot_minutes // SIM_SLOT_MINUTES))
        self.risk_cache = {}
        self.model = None
        self.pre = {}
        for name in ["expected", "mu", "pi", "theta", "v_hat"]:
            path = p / f"test_predictions_{name}.npy"
            if path.exists():
                self.pre[name] = np.load(path).astype(np.float32)
        candidates = [Path(cfg.model_path), Path(cfg.root_dir) / "best_cc_stmt_model.pth", Path(cfg.proc_dir).parent / "prediction" / "models" / "best_CC_STMT_model.pth", Path(cfg.root_dir) / "models" / cfg.city / "best_CC_STMT_model.pth"]
        model_path = next((x for x in candidates if x.exists()), None)
        if model_path is not None:
            self.model = CC_STMT(num_nodes=self.num_nodes, input_dim=5, hidden_dim=64, num_layers=2, nhead=8, seq_len=SEQ_LEN).to(device)
            self.model.load_state_dict(safe_torch_load(model_path, device))
            self.model.eval()
            log(f"loaded model {model_path}")
        elif {"expected", "pi"}.issubset(self.pre.keys()):
            log("using precomputed prediction arrays")
        else:
            raise FileNotFoundError(f"missing model and prediction arrays under {p}")

    def slice_pad(self, x, idx, mean, std):
        start = max(0, int(idx) - SEQ_LEN)
        out = (x[start:int(idx)] - mean) / std
        if len(out) < SEQ_LEN:
            pad = SEQ_LEN - len(out)
            out = np.pad(out, [(pad, 0)] + [(0, 0)] * (out.ndim - 1), mode="constant")
        return out

    def realtime(self, env_idx):
        env_idx = int(np.clip(env_idx, 1, len(self.x_order)))
        if self.model is not None:
            xo = torch.FloatTensor(self.slice_pad(self.x_order, env_idx, self.m_order, self.s_order)).unsqueeze(0).to(self.device)
            xs = torch.FloatTensor(self.slice_pad(self.x_speed, env_idx, self.m_speed, self.s_speed)).unsqueeze(0).to(self.device)
            xe = torch.FloatTensor(self.slice_pad(self.x_enroute, env_idx, self.m_enroute, self.s_enroute)).unsqueeze(0).to(self.device)
            xw = torch.FloatTensor(self.slice_pad(self.x_weather, env_idx, self.m_w, self.s_w)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pi, mu, theta, _ = self.model(xo, xs, xe, xw, self.adj_spatial, self.adj_semantic)
            mu = (mu * self.s_order + self.m_order).squeeze(0).detach().cpu().numpy()
            pi = pi.squeeze(0).detach().cpu().numpy()
            theta = theta.squeeze(0).detach().cpu().numpy()
            return np.clip(safe_to_1d(mu, self.num_nodes), 0.0, None), np.clip(safe_to_1d(theta, self.num_nodes), 1e-4, None), np.clip(safe_to_1d(pi, self.num_nodes), 1e-4, 1.0 - 1e-4)
        k = int(env_idx - self.val_end - SEQ_LEN)
        if "mu" in self.pre:
            arr = self.pre["mu"]
        else:
            arr = self.pre["expected"]
        k = int(np.clip(k, 0, len(arr) - 1))
        mu = safe_to_1d(arr[k], self.num_nodes)
        pi = safe_to_1d(self.pre["pi"][k], self.num_nodes)
        theta = safe_to_1d(self.pre["theta"][k], self.num_nodes) if "theta" in self.pre else np.ones(self.num_nodes, dtype=np.float32)
        return np.clip(mu, 0.0, None), np.clip(theta, 1e-4, None), np.clip(pi, 1e-4, 1.0 - 1e-4)

    def realized_risk(self, env_idx, horizon=2, metric="opportunity"):
        key = (int(env_idx), int(horizon), str(metric))
        if key in self.risk_cache:
            return self.risk_cache[key]
        start = int(np.clip(env_idx + 1, 0, len(self.x_order) - 1))
        end = int(np.clip(start + max(1, int(horizon)), start + 1, len(self.x_order)))
        future = self.x_order[start:end, :, 0].sum(axis=0).astype(np.float32)
        if metric == "imbalance":
            supply = self.x_enroute[start:end, :, 0].sum(axis=0).astype(np.float32)
            x = np.maximum(supply - future, 0.0)
            risk = np.clip(x / (np.nanpercentile(x, 95) + 1e-8), 0.0, 1.0)
        elif metric == "hybrid":
            supply = self.x_enroute[start:end, :, 0].sum(axis=0).astype(np.float32)
            x = np.maximum(supply - future, 0.0)
            imb = np.clip(x / (np.nanpercentile(x, 95) + 1e-8), 0.0, 1.0)
            opp = 1.0 - np.clip(future / (np.nanpercentile(future, 95) + 1e-8), 0.0, 1.0)
            risk = 0.5 * imb + 0.5 * opp
        else:
            risk = 1.0 - np.clip(future / (np.nanpercentile(future, 95) + 1e-8), 0.0, 1.0)
        self.risk_cache[key] = risk.astype(np.float32)
        return self.risk_cache[key]

    def future_opportunity(self, sim_slot, horizon=2):
        env_idx = int(np.clip(sim_slot // self.pred_step_ratio, 0, len(self.x_order) - 1))
        start = int(np.clip(env_idx + 1, 0, len(self.x_order) - 1))
        end = int(min(len(self.x_order), start + max(1, int(horizon))))
        return safe_to_1d(self.x_order[start:end].sum(axis=0), self.num_nodes), env_idx

class SafeUOTDispatcher:
    def __init__(self, name, variant, dist_matrix, weights, scales, max_pickup):
        self.name = name
        self.variant = variant
        self.dist_matrix = dist_matrix.astype(np.float32)
        self.weights = weights
        self.scales = scales
        self.max_pickup = float(max_pickup)

    def sinkhorn(self, cost, mask, epsilon=0.08, tau=0.85, max_iter=50):
        c = np.where(mask, cost, 1e3).astype(np.float32)
        c = c - np.nanmin(c)
        c = c / (np.nanpercentile(c[mask], 90) + 1e-8) if np.any(mask) else c
        k = np.exp(-c / max(epsilon, 1e-6)) * mask.astype(np.float32)
        a = np.ones(cost.shape[0], dtype=np.float32) / max(cost.shape[0], 1)
        b = np.ones(cost.shape[1], dtype=np.float32) / max(cost.shape[1], 1)
        u = np.ones_like(a)
        v = np.ones_like(b)
        power = tau / (tau + epsilon)
        for _ in range(max_iter):
            u = (a / (k @ v + 1e-8)) ** power
            v = (b / (k.T @ u + 1e-8)) ** power
        return (u[:, None] * k) * v[None, :]

    def build_cost(self, vehicles, orders, mu, pi, current_slot):
        nv, no = len(vehicles), len(orders)
        cost = np.full((nv, no), 1e3, dtype=np.float32)
        mask = np.zeros((nv, no), dtype=bool)
        if nv == 0 or no == 0:
            return cost, mask
        veh_z = np.array([v.current_zone for v in vehicles], dtype=int)
        pu_z = np.array([o.pickup_zone for o in orders], dtype=int)
        do_z = np.array([o.dropoff_zone for o in orders], dtype=int)
        dist = self.dist_matrix[np.ix_(veh_z, pu_z)]
        mask = np.isfinite(dist) & (dist <= self.max_pickup)
        if not np.any(mask):
            return cost, mask
        rev = np.array([o.revenue for o in orders], dtype=np.float32)[None, :]
        wait = np.array([wait_minutes(o, current_slot) for o in orders], dtype=np.float32)[None, :]
        dest_mu = np.clip(mu[do_z], 0.0, None)
        dest_pi = np.clip(pi[do_z], 0.0, 1.0)
        value = np.log1p((1.0 - dest_pi) * dest_mu)
        if self.variant.adjusted_value:
            value = value * np.power(1.0 - np.clip(dest_pi, 0.0, 1.0), self.variant.alpha)
        risk = np.power(dest_pi, self.variant.beta)
        dist_n = norm_by_scale(dist, self.scales["dist"])
        rev_n = norm_by_scale(rev, self.scales["rev"])
        wait_n = norm_by_scale(wait, self.scales["wait"])
        value_n = norm_by_scale(value[None, :], self.scales["value"])
        risk_n = np.clip(risk[None, :], 0.0, 1.0).astype(np.float32)
        cost = self.weights["dist"] * dist_n - self.weights["rev"] * rev_n - self.weights["wait"] * wait_n
        if self.variant.use_value:
            cost -= self.weights["dest_val"] * value_n
        if self.variant.use_risk:
            cost += self.weights["dest_risk"] * risk_n
        cost = np.where(mask, cost, 1e3).astype(np.float32)
        return cost, mask

    def solve(self, vehicles, orders, mu, pi, current_slot):
        cost, mask = self.build_cost(vehicles, orders, mu, pi, current_slot)
        if cost.size == 0 or not np.any(mask):
            return []
        prior = self.sinkhorn(cost, mask)
        final_cost = cost - UOT_GAMMA * minmax_masked(prior, mask)
        final_cost = np.where(mask, final_cost, 1e6)
        rows, cols = linear_sum_assignment(final_cost)
        out = []
        for r, c in zip(rows, cols):
            if mask[r, c]:
                out.append((int(r), int(c), float(cost[r, c])))
        return out

def load_zone_mapping(cfg, num_nodes):
    p = Path(cfg.proc_dir)
    for name in ["zone_mapping.csv", "zones.csv"]:
        f = p / name
        if f.exists():
            z = pd.read_csv(f)
            cols = z.columns.tolist()
            idx_col = find_col(cols, ["zone_idx", "idx", "node", "node_idx"], False)
            raw_col = find_col(cols, ["zone_raw_id", "LocationID", "geoid10", "community_area", "community"], False)
            if idx_col is not None and raw_col is not None:
                return {str(x): int(i) for x, i in zip(z[raw_col], z[idx_col])}
    if cfg.city_type == "nyc":
        try:
            import geopandas as gpd
            shp = Path(cfg.raw_dir) / "taxi_zones.shp"
            zones = gpd.read_file(shp).sort_values("LocationID").reset_index(drop=True)
            return {str(int(x)): int(i) for i, x in enumerate(zones["LocationID"])}
        except Exception:
            return {str(i + 1): i for i in range(num_nodes)}
    if cfg.city_type == "chicago_tract":
        try:
            import geopandas as gpd
            shp = Path(cfg.raw_dir) / "geo_export_bbf57716-e851-4380-a37c-24fbe460c069.shp"
            zones = gpd.read_file(shp).to_crs("EPSG:4326")
            id_col = "geoid10" if "geoid10" in zones.columns else zones.columns[0]
            zones[id_col] = zones[id_col].astype(str).str.strip().str.replace(".0", "", regex=False)
            loop = gpd.points_from_xy([-87.6231], [41.8818], crs="EPSG:4326").to_crs("EPSG:32616")[0]
            zp = zones.to_crs("EPSG:32616")
            zones = zones[zp.geometry.centroid.distance(loop) <= 8000].copy().sort_values(id_col).reset_index(drop=True)
            return {str(x): int(i) for i, x in enumerate(zones[id_col])}
        except Exception:
            return {}
    return {str(i + 1): i for i in range(num_nodes)}

def load_standard_orders(path, cfg):
    df = pd.read_csv(path, low_memory=False)
    base_time = pd.to_datetime(cfg.base_time)
    cols = df.columns.tolist()
    start_col = find_col(cols, ["trip_start_timestamp", "pickup_datetime", "pickup_time", "start_time"], True)
    end_col = find_col(cols, ["trip_end_timestamp", "dropoff_datetime", "dropoff_time", "end_time"], False)
    pu_col = find_col(cols, ["PULocationID", "pu_idx", "pickup_zone", "pickup_idx"], True)
    do_col = find_col(cols, ["DOLocationID", "do_idx", "dropoff_zone", "dropoff_idx"], True)
    miles_col = find_col(cols, ["trip_miles", "miles", "Trip Miles"], False)
    fare_col = find_col(cols, ["base_passenger_fare", "trip_total", "fare", "revenue"], False)
    seconds_col = find_col(cols, ["trip_seconds", "duration_sec", "duration_seconds"], False)
    pickup_slot_col = find_col(cols, ["pickup_slot_5min", "start_slot", "creation_slot_5min"], False)
    duration_col = find_col(cols, ["duration_slots_5min", "duration_slots"], False)
    out = pd.DataFrame()
    out["pickup_time"] = pd.to_datetime(df[start_col], errors="coerce")
    out["dropoff_time"] = pd.to_datetime(df[end_col], errors="coerce") if end_col else pd.NaT
    out["pu_idx"] = pd.to_numeric(df[pu_col], errors="coerce")
    out["do_idx"] = pd.to_numeric(df[do_col], errors="coerce")
    out["miles"] = pd.to_numeric(df[miles_col], errors="coerce") if miles_col else 1.0
    out["fare"] = clean_money(df[fare_col]) if fare_col else out["miles"] * 2.5
    out["trip_seconds"] = pd.to_numeric(df[seconds_col], errors="coerce") if seconds_col else (out["dropoff_time"] - out["pickup_time"]).dt.total_seconds()
    out["pickup_slot_5min"] = pd.to_numeric(df[pickup_slot_col], errors="coerce") if pickup_slot_col else np.floor((out["pickup_time"] - base_time).dt.total_seconds() / 300.0)
    out["duration_slots_5min"] = pd.to_numeric(df[duration_col], errors="coerce") if duration_col else np.ceil(out["trip_seconds"] / 300.0)
    return clean_order_frame(out)

def load_raw_nyc_orders(cfg, num_nodes):
    files = sorted(Path(cfg.raw_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no NYC parquet files in {cfg.raw_dir}")
    zone_map = load_zone_mapping(cfg, num_nodes)
    parts = []
    for f in files:
        try:
            import pyarrow.parquet as pq
            cols = pq.ParquetFile(f).schema.names
        except Exception:
            cols = pd.read_parquet(f).columns.tolist()
        start_col = find_col(cols, ["pickup_datetime", "request_datetime", "Pickup Datetime"], True)
        end_col = find_col(cols, ["dropoff_datetime", "Dropoff Datetime"], True)
        use = [c for c in [start_col, end_col, find_col(cols, ["PULocationID"], True), find_col(cols, ["DOLocationID"], True), find_col(cols, ["trip_miles"], False), find_col(cols, ["base_passenger_fare", "fare"], False)] if c is not None]
        parts.append(pd.read_parquet(f, columns=list(dict.fromkeys(use))))
    df = pd.concat(parts, ignore_index=True)
    cols = df.columns.tolist()
    start_col = find_col(cols, ["pickup_datetime"], True)
    end_col = find_col(cols, ["dropoff_datetime"], True)
    pu_col = find_col(cols, ["PULocationID"], True)
    do_col = find_col(cols, ["DOLocationID"], True)
    miles_col = find_col(cols, ["trip_miles"], False)
    fare_col = find_col(cols, ["base_passenger_fare", "fare"], False)
    out = pd.DataFrame()
    out["pickup_time"] = pd.to_datetime(df[start_col], errors="coerce")
    out["dropoff_time"] = pd.to_datetime(df[end_col], errors="coerce")
    out["pu_idx"] = df[pu_col].astype(str).map(zone_map)
    out["do_idx"] = df[do_col].astype(str).map(zone_map)
    out["miles"] = pd.to_numeric(df[miles_col], errors="coerce") if miles_col else 1.0
    out["fare"] = clean_money(df[fare_col]) if fare_col else out["miles"] * 2.5
    out["trip_seconds"] = (out["dropoff_time"] - out["pickup_time"]).dt.total_seconds()
    base_time = pd.to_datetime(cfg.base_time)
    out["pickup_slot_5min"] = np.floor((out["pickup_time"] - base_time).dt.total_seconds() / 300.0)
    out["duration_slots_5min"] = np.ceil(out["trip_seconds"] / 300.0)
    return clean_order_frame(out)

def load_raw_chicago_orders(cfg, num_nodes):
    files = sorted(Path(cfg.raw_dir).glob("Transportation_Network_Providers*.csv"))
    if not files and cfg.city_type == "chicago_area":
        files = sorted((Path(cfg.root_dir) / "data" / "chiago").glob("Transportation_Network_Providers*.csv"))
    if not files:
        raise FileNotFoundError(f"no Chicago TNP csv files in {cfg.raw_dir}")
    zone_map = load_zone_mapping(cfg, num_nodes)
    parts = []
    for f in files:
        cols = pd.read_csv(f, nrows=0).columns.tolist()
        targets = {"trip_start_timestamp", "trip_end_timestamp", "trip_seconds", "trip_miles", "fare", "trip_total", "pickup_census_tract", "dropoff_census_tract", "pickup_community_area", "dropoff_community_area"}
        use = [c for c in cols if norm_col(c) in targets]
        part = pd.read_csv(f, usecols=use, low_memory=False)
        part.rename(columns={c: norm_col(c) for c in part.columns}, inplace=True)
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    if cfg.city_type == "chicago_area":
        pu_key = "pickup_community_area"
        do_key = "dropoff_community_area"
    else:
        pu_key = "pickup_census_tract"
        do_key = "dropoff_census_tract"
    out = pd.DataFrame()
    out["pickup_time"] = pd.to_datetime(df["trip_start_timestamp"], errors="coerce")
    out["dropoff_time"] = pd.to_datetime(df["trip_end_timestamp"], errors="coerce")
    out["pu_idx"] = df[pu_key].apply(clean_area_id).map(zone_map)
    out["do_idx"] = df[do_key].apply(clean_area_id).map(zone_map)
    out["miles"] = pd.to_numeric(df.get("trip_miles", np.nan), errors="coerce")
    fare_col = "trip_total" if "trip_total" in df.columns else "fare" if "fare" in df.columns else None
    out["fare"] = clean_money(df[fare_col]) if fare_col else out["miles"] * 2.5
    out["trip_seconds"] = pd.to_numeric(df.get("trip_seconds", np.nan), errors="coerce")
    missing = out["trip_seconds"].isna()
    out.loc[missing, "trip_seconds"] = (out.loc[missing, "dropoff_time"] - out.loc[missing, "pickup_time"]).dt.total_seconds()
    base_time = pd.to_datetime(cfg.base_time)
    out["pickup_slot_5min"] = np.floor((out["pickup_time"] - base_time).dt.total_seconds() / 300.0)
    out["duration_slots_5min"] = np.ceil(out["trip_seconds"] / 300.0)
    return clean_order_frame(out)

def clean_order_frame(out):
    out = out.dropna(subset=["pickup_time", "pu_idx", "do_idx", "pickup_slot_5min", "duration_slots_5min", "fare", "miles"]).copy()
    out = out[(out["fare"] > 0) & (out["miles"] > 0) & (out["duration_slots_5min"] >= 1)].copy()
    out["pu_idx"] = out["pu_idx"].astype(int)
    out["do_idx"] = out["do_idx"].astype(int)
    out["pickup_slot_5min"] = out["pickup_slot_5min"].astype(int)
    out["duration_slots_5min"] = out["duration_slots_5min"].astype(int)
    out = out.sort_values("pickup_time").reset_index(drop=True)
    return out

def load_orders(cfg, num_nodes):
    p = Path(cfg.proc_dir)
    candidates = ["dispatch_orders.csv", "clean_orders.csv", "chicago_bigfull_clean_orders.csv", "orders.csv"]
    for name in candidates:
        f = p / name
        if f.exists():
            out = load_standard_orders(f, cfg)
            out = out[(out["pu_idx"] >= 0) & (out["pu_idx"] < num_nodes) & (out["do_idx"] >= 0) & (out["do_idx"] < num_nodes)].copy()
            return out
    if cfg.city_type == "nyc":
        out = load_raw_nyc_orders(cfg, num_nodes)
    else:
        out = load_raw_chicago_orders(cfg, num_nodes)
    out = out[(out["pu_idx"] >= 0) & (out["pu_idx"] < num_nodes) & (out["do_idx"] >= 0) & (out["do_idx"] < num_nodes)].copy()
    save_path = p / "dispatch_orders.csv"
    out.to_csv(save_path, index=False)
    log(f"saved {save_path}")
    return out

def build_orders(df, seed, advance_prob):
    rng = np.random.default_rng(seed + int(advance_prob * 1000))
    orders = []
    for i, row in df.reset_index(drop=True).iterrows():
        start = int(row["pickup_slot_5min"])
        dur = max(1, int(row["duration_slots_5min"]))
        is_adv = bool((float(row["miles"]) > 8.0 or rng.random() < advance_prob) if advance_prob > 0 else False)
        lead = int(rng.integers(6, 24)) if is_adv else 0
        creation = max(0, start - lead) if is_adv else start
        latest = start + (ADVANCE_SERVICE_TOLERANCE_SLOTS if is_adv else MAX_WAIT_SLOTS)
        orders.append(OrderState(int(i), int(row["pu_idx"]), int(row["do_idx"]), start, dur, float(row["fare"]), float(row["miles"]), creation, is_adv, latest, lead))
    orders.sort(key=lambda x: (x.creation_slot, x.start_slot, x.id))
    return orders

def sample_orders(df, max_orders, seed):
    original = len(df)
    if max_orders is None or max_orders <= 0 or original <= max_orders:
        return df.sort_values("pickup_time").copy(), 1.0, original
    ratio = max_orders / float(original)
    tmp = df.copy()
    tmp["_hour"] = tmp["pickup_time"].dt.hour.astype(str)
    tmp["_pu"] = tmp["pu_idx"].astype(str)
    tmp["_strata"] = tmp["_hour"] + "_" + tmp["_pu"]
    rng = np.random.default_rng(seed)
    parts = []
    for _, part in tmp.groupby("_strata", sort=False):
        n = len(part)
        k = int(round(n * ratio))
        if k <= 0 and rng.random() < min(1.0, n * ratio):
            k = 1
        if k > 0:
            parts.append(part.sample(n=min(k, n), random_state=seed))
    out = pd.concat(parts, axis=0) if parts else tmp.sample(n=max_orders, random_state=seed)
    if len(out) > max_orders:
        out = out.sample(n=max_orders, random_state=seed)
    return out.drop(columns=["_hour", "_pu", "_strata"], errors="ignore").sort_values("pickup_time").copy(), len(out) / float(original), original

def initial_vehicle_zones(df_before, n, num_zones, seed):
    rng = np.random.default_rng(seed)
    if len(df_before) > 0:
        vals = df_before["do_idx"].dropna().astype(int).values
        vals = vals[(vals >= 0) & (vals < num_zones)]
        if len(vals) > 0:
            return rng.choice(vals, size=n, replace=True)
    return rng.integers(0, num_zones, size=n)

def compute_scales(orders, dist_matrix, max_pickup):
    sample = orders[:min(len(orders), 50000)]
    rev = np.array([o.revenue for o in sample], dtype=np.float32)
    wait = np.array([MAX_WAIT_SLOTS * SIM_SLOT_MINUTES for _ in sample], dtype=np.float32)
    value = np.array([1.0 for _ in sample], dtype=np.float32)
    valid = dist_matrix[np.isfinite(dist_matrix) & (dist_matrix <= max_pickup)]
    return {
        "dist": float(np.nanpercentile(valid, 95) if valid.size else max_pickup),
        "rev": float(np.nanpercentile(rev, 95) if rev.size else 1.0),
        "wait": float(np.nanpercentile(wait, 95) if wait.size else 1.0),
        "value": float(np.nanpercentile(value, 95) if value.size else 1.0)
    }

class Simulation:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = Predictor(cfg, self.device)
        self.dist_matrix = np.load(Path(cfg.proc_dir) / "dist_matrix.npy").astype(np.float32)
        self.num_zones = int(self.dist_matrix.shape[0])
        if self.num_zones != self.predictor.num_nodes:
            raise RuntimeError("node count mismatch")
        self.orders_df = load_orders(cfg, self.num_zones)
        self.pred_step_ratio = self.predictor.pred_step_ratio
        self.weights = WEIGHT_PROFILES[args.weight_profile]

    def variants(self):
        return [
            Variant("Base-UOT", False, False, False, 0.0, 0.0),
            Variant("Mu-only", True, False, False, 0.0, 0.0),
            Variant("Pi-only", False, True, False, 0.0, 1.0),
            Variant("Mu-plus-Pi", True, True, False, 0.0, 1.0),
            Variant("DP-OTM", True, True, True, self.args.proposed_alpha, self.args.proposed_beta)
        ]

    def run(self):
        set_seed(self.args.seed)
        val_start = self.predictor.val_end * self.pred_step_ratio
        sim_slots = int(self.args.test_days * 24 * 60 / SIM_SLOT_MINUTES)
        clock_end = val_start + sim_slots
        test_df = self.orders_df[(self.orders_df["pickup_slot_5min"] >= val_start) & (self.orders_df["pickup_slot_5min"] < clock_end)].copy()
        if len(test_df) == 0:
            raise RuntimeError("empty dispatch test window")
        sampled, ratio, original = sample_orders(test_df, self.args.max_orders, self.args.seed)
        orders_pure = build_orders(sampled, self.args.seed, 0.0)
        orders_mixed = build_orders(sampled, self.args.seed, self.args.advance_prob)
        before = self.orders_df[self.orders_df["pickup_slot_5min"] < val_start]
        max_pickup = self.args.max_pickup_dist if self.args.max_pickup_dist is not None else self.cfg.default_max_pickup_dist / math.sqrt(max(ratio, 1e-8))
        scales = compute_scales(orders_mixed, self.dist_matrix, max_pickup)
        rows = []
        scenarios = []
        if self.args.scenario in {"pure", "both"}:
            scenarios.append(("Pure On-Demand", orders_pure))
        if self.args.scenario in {"mixed", "both"}:
            scenarios.append(("Mixed Advance", orders_mixed))
        for scenario, orders in scenarios:
            for fleet in self.args.fleets:
                init_z = initial_vehicle_zones(before, fleet, self.num_zones, self.args.seed + fleet)
                for variant in self.variants():
                    row = self.run_variant(scenario, orders, fleet, init_z, variant, val_start, clock_end, max_pickup, scales, original, ratio)
                    rows.append(row)
                    log(f"{self.cfg.city} {scenario} fleet={fleet} {variant.name} reject={row['Reject(%)']:.3f} risk={row['Risk']:.4f} pdo={row['PDO_H']:.3f}")
        df = pd.DataFrame(rows)
        out = ensure_dir(self.cfg.out_dir) / f"{self.cfg.city}_dispatch_seed{self.args.seed}.csv"
        df.to_csv(out, index=False)
        summary = self.summary(df)
        summary.to_csv(ensure_dir(self.cfg.out_dir) / f"{self.cfg.city}_dispatch_summary_seed{self.args.seed}.csv", index=False)
        log(f"saved {out}")
        return df

    def run_variant(self, scenario, orders, fleet, init_z, variant, clock_start, clock_end, max_pickup, scales, original, ratio):
        vehicles = [VehicleState(i, int(z)) for i, z in enumerate(init_z)]
        dispatcher = SafeUOTDispatcher(variant.name, variant, self.dist_matrix, self.weights, scales, max_pickup)
        pending = []
        idx = 0
        served = 0
        rejected = 0
        rev = 0.0
        empty = 0.0
        wait_sum = 0.0
        risk_sum = 0.0
        risk_count = 0
        pdo_sum = 0.0
        pdo_count = 0
        der_low = 0
        der_zero = 0
        adv_served = 0
        ond_served = 0
        adv_rejected = 0
        ond_rejected = 0
        t0 = time.time()
        orders = list(orders)
        while idx < len(orders) and orders[idx].creation_slot < clock_start:
            idx += 1
        for t in range(int(clock_start), int(clock_end)):
            while idx < len(orders) and orders[idx].creation_slot <= t:
                pending.append(orders[idx])
                idx += 1
            still = []
            for o in pending:
                if order_expired(o, t):
                    rejected += 1
                    if o.is_advance:
                        adv_rejected += 1
                    else:
                        ond_rejected += 1
                else:
                    still.append(o)
            pending = still
            active_orders = [o for o in pending if order_dispatchable(o, t)]
            available = [v for v in vehicles if v.available_time <= t]
            if not active_orders or not available:
                continue
            env_idx = int(t // self.pred_step_ratio)
            mu, theta, pi = self.predictor.realtime(env_idx)
            matches = dispatcher.solve(available, active_orders, mu, pi, t)
            if not matches:
                continue
            used_v = set()
            used_o = set()
            for vi, oi, _ in matches:
                if vi in used_v or oi in used_o:
                    continue
                v = available[vi]
                o = active_orders[oi]
                dist = float(self.dist_matrix[v.current_zone, o.pickup_zone])
                if not np.isfinite(dist) or dist > max_pickup:
                    continue
                service_start, service_end = v.add_order(o, dist, t)
                used_v.add(vi)
                used_o.add(oi)
                served += 1
                rev += o.revenue
                empty += dist
                wait_sum += realized_wait_minutes(o, service_start)
                if o.is_advance:
                    adv_served += 1
                else:
                    ond_served += 1
                r = self.predictor.realized_risk(int(service_end // self.pred_step_ratio), self.args.risk_horizon_steps, self.args.risk_metric)
                risk_sum += float(r[o.dropoff_zone])
                risk_count += 1
                future, _ = self.predictor.future_opportunity(service_end, self.args.risk_horizon_steps)
                opp = float(future[o.dropoff_zone])
                pdo_sum += opp
                pdo_count += 1
                low_thr = float(np.nanpercentile(future, 20))
                der_low += int(opp <= low_thr)
                der_zero += int(opp <= 1e-8)
            if used_o:
                remove_ids = {active_orders[i].id for i in used_o}
                pending = [o for o in pending if o.id not in remove_ids]
        rejected += len(pending)
        for o in pending:
            if o.is_advance:
                adv_rejected += 1
            else:
                ond_rejected += 1
        total = served + rejected
        return {
            "City": self.cfg.city,
            "Scenario": scenario,
            "Seed": self.args.seed,
            "Fleet": fleet,
            "Algorithm": variant.name,
            "OriginalOrders": int(original),
            "UsedOrders": int(len(orders)),
            "SampleRatio": float(ratio),
            "Served": int(served),
            "Rejected": int(rejected),
            "Reject(%)": float(100.0 * rejected / max(total, 1)),
            "Rev($)": float(rev),
            "EmptyMiles": float(empty),
            "Risk": float(risk_sum / max(risk_count, 1)),
            "Wait(m)": float(wait_sum / max(served, 1)),
            "DERZero(%)": float(100.0 * der_zero / max(pdo_count, 1)),
            "DERLow20(%)": float(100.0 * der_low / max(pdo_count, 1)),
            "PDO_H": float(pdo_sum / max(pdo_count, 1)),
            "AdvDispatched": int(adv_served),
            "OnDemandDispatched": int(ond_served),
            "AdvRejected": int(adv_rejected),
            "OnDemandRejected": int(ond_rejected),
            "UseDemandValue": bool(variant.use_value),
            "UseRiskPenalty": bool(variant.use_risk),
            "UseRiskAdjustedValue": bool(variant.adjusted_value),
            "RiskValueAlpha": float(variant.alpha),
            "RiskPenaltyBeta": float(variant.beta),
            "MaxPickupDist": float(max_pickup),
            "Time(s)": float(time.time() - t0)
        }

    def summary(self, df):
        keys = ["City", "Scenario", "Fleet", "Algorithm"]
        metrics = ["Reject(%)", "Rev($)", "EmptyMiles", "Risk", "Wait(m)", "DERZero(%)", "DERLow20(%)", "PDO_H", "Time(s)"]
        rows = []
        for key, part in df.groupby(keys, dropna=False):
            row = dict(zip(keys, key))
            for m in metrics:
                row[m] = float(part[m].mean())
                row[m + "_std"] = float(part[m].std(ddof=0))
            rows.append(row)
        return pd.DataFrame(rows)

def run_cli(default_cfg: CityConfig):
    p = argparse.ArgumentParser()
    p.add_argument("--proc_dir", default=default_cfg.proc_dir)
    p.add_argument("--raw_dir", default=default_cfg.raw_dir)
    p.add_argument("--out_dir", default=default_cfg.out_dir)
    p.add_argument("--model_path", default=default_cfg.model_path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fleets", nargs="*", type=int, default=list(default_cfg.default_fleets))
    p.add_argument("--max_orders", type=int, default=default_cfg.default_max_orders)
    p.add_argument("--test_days", type=int, default=1)
    p.add_argument("--scenario", choices=["pure", "mixed", "both"], default="both")
    p.add_argument("--advance_prob", type=float, default=0.30)
    p.add_argument("--risk_horizon_steps", type=int, default=2)
    p.add_argument("--max_pickup_dist", type=float, default=None)
    p.add_argument("--weight_profile", choices=sorted(WEIGHT_PROFILES.keys()), default="paper")
    p.add_argument("--risk_metric", choices=["opportunity", "imbalance", "hybrid"], default="opportunity")
    p.add_argument("--proposed_alpha", type=float, default=0.10)
    p.add_argument("--proposed_beta", type=float, default=1.20)
    args = p.parse_args()
    cfg = CityConfig(default_cfg.city, default_cfg.root_dir, args.proc_dir, args.raw_dir, args.out_dir, args.model_path, default_cfg.base_time, default_cfg.pred_slot_minutes, default_cfg.city_type, tuple(args.fleets), args.max_orders, default_cfg.default_max_pickup_dist)
    ensure_dir(cfg.out_dir)
    sim = Simulation(cfg, args)
    sim.run()
