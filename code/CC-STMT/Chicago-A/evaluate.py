import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(COMMON_DIR))

from model import CC_STMT

DEFAULT_PROCESSED_DIR = "/root/autodl-tmp/new/data/chiago/processed"
DEFAULT_MODEL_DIR = "/root/autodl-tmp/new/models/Chicago-A"
DEFAULT_RESULT_DIR = "/root/autodl-tmp/new/results/Chicago-A"

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def align_weather(weather, total_slots, num_nodes):
    if weather.ndim == 2:
        weather = np.expand_dims(weather, axis=1).repeat(num_nodes, axis=1)
    elif weather.ndim == 3 and weather.shape[1] == 1:
        weather = np.repeat(weather, num_nodes, axis=1)
    if weather.shape[0] != total_slots:
        weather = weather[:min(weather.shape[0], total_slots)]
    return weather.astype(np.float32)

def crop_arrays(arrays):
    n = min(v.shape[0] for v in arrays.values() if isinstance(v, np.ndarray) and v.ndim >= 1)
    for k in list(arrays.keys()):
        if isinstance(arrays[k], np.ndarray) and arrays[k].ndim >= 1 and arrays[k].shape[0] != n:
            arrays[k] = arrays[k][:n]
    return arrays

def load_arrays(processed_dir):
    p = Path(processed_dir)
    demand = np.load(p / "demand_tensor_full.npy").astype(np.float32)
    speed = np.load(p / "speed_tensor_full.npy").astype(np.float32)
    enroute = np.load(p / "enroute_tensor_full.npy").astype(np.float32)
    weather = np.load(p / "weather_tensor.npy").astype(np.float32)
    adj_spatial = np.load(p / "adj_spatial.npy").astype(np.float32)
    adj_semantic = np.load(p / "adj_semantic.npy").astype(np.float32)
    weather = align_weather(weather, demand.shape[0], demand.shape[1])
    return crop_arrays({"demand": demand, "speed": speed, "enroute": enroute, "weather": weather, "adj_spatial": adj_spatial, "adj_semantic": adj_semantic})

def normalize_with_stats(arrays, stats):
    norm = {}
    norm["demand"] = (arrays["demand"] - float(stats["mean_order"])) / float(stats["std_order"])
    norm["speed"] = (arrays["speed"] - float(stats["mean_speed"])) / float(stats["std_speed"])
    norm["enroute"] = (arrays["enroute"] - float(stats["mean_enroute"])) / float(stats["std_enroute"])
    norm["weather"] = (arrays["weather"] - stats["mean_weather"]) / stats["std_weather"]
    return norm

class RideHailingDataset(Dataset):
    def __init__(self, norm, raw_demand, raw_speed, start, end, seq_len):
        self.x_order = norm["demand"][start:end]
        self.x_speed = norm["speed"][start:end]
        self.x_enroute = norm["enroute"][start:end]
        self.x_weather = norm["weather"][start:end]
        self.raw_demand = raw_demand[start:end]
        self.raw_speed = raw_speed[start:end]
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.x_order) - self.seq_len)

    def __getitem__(self, idx):
        xo = self.x_order[idx:idx + self.seq_len]
        xs = self.x_speed[idx:idx + self.seq_len]
        xe = self.x_enroute[idx:idx + self.seq_len]
        xw = self.x_weather[idx:idx + self.seq_len]
        yo = self.raw_demand[idx + self.seq_len, :, 0]
        ys = self.raw_speed[idx + self.seq_len, :, 0]
        return torch.FloatTensor(xo), torch.FloatTensor(xs), torch.FloatTensor(xe), torch.FloatTensor(xw), torch.FloatTensor(yo), torch.FloatTensor(ys)

def calc_metrics(pred, truth):
    pred = pred.reshape(-1)
    truth = truth.reshape(-1)
    mae = float(np.mean(np.abs(pred - truth)))
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    mask = truth > 0
    mape = float(np.mean(np.abs((truth[mask] - pred[mask]) / truth[mask])) * 100.0) if np.any(mask) else 0.0
    return {"MAE": mae, "RMSE": rmse, "MAPE_nonzero_percent": mape}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    ensure_dir(args.result_dir)
    arrays = load_arrays(args.processed_dir)
    stats = dict(np.load(Path(args.model_dir) / "norm_stats.npz", allow_pickle=True))
    seq_len = int(stats["seq_len"])
    val_end = int(stats["val_end"])
    norm = normalize_with_stats(arrays, stats)
    dataset = RideHailingDataset(norm, arrays["demand"], arrays["speed"], val_end, arrays["demand"].shape[0], seq_len)
    if len(dataset) == 0:
        raise RuntimeError("The test dataset is empty")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 3 + arrays["weather"].shape[-1]
    model = CC_STMT(num_nodes=arrays["demand"].shape[1], input_dim=input_dim, hidden_dim=64, num_layers=2, nhead=8, seq_len=seq_len).to(device)
    state = torch.load(Path(args.model_dir) / "best_cc_stmt_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    adj_spatial = torch.FloatTensor(arrays["adj_spatial"]).to(device)
    adj_semantic = torch.FloatTensor(arrays["adj_semantic"]).to(device)
    mean_order = float(stats["mean_order"])
    std_order = float(stats["std_order"])
    outputs = {"expected": [], "truth": [], "pi": [], "mu": [], "theta": [], "v_hat": []}
    with torch.no_grad():
        for xo, xs, xe, xw, yo, ys in loader:
            xo = xo.to(device)
            xs = xs.to(device)
            xe = xe.to(device)
            xw = xw.to(device)
            pi, mu, theta, v_hat = model(xo, xs, xe, xw, adj_spatial, adj_semantic)
            mu = mu * std_order + mean_order
            expected = (1.0 - pi) * mu
            outputs["expected"].append(expected.cpu().numpy())
            outputs["truth"].append(yo.numpy())
            outputs["pi"].append(pi.cpu().numpy())
            outputs["mu"].append(mu.cpu().numpy())
            outputs["theta"].append(theta.cpu().numpy())
            outputs["v_hat"].append(v_hat.cpu().numpy())
    for k in outputs:
        outputs[k] = np.concatenate(outputs[k], axis=0)
    p = Path(args.processed_dir)
    r = Path(args.result_dir)
    np.save(p / "test_predictions_expected.npy", outputs["expected"])
    np.save(p / "test_predictions_mu.npy", outputs["mu"])
    np.save(p / "test_predictions_pi.npy", outputs["pi"])
    np.save(p / "test_predictions_theta.npy", outputs["theta"])
    np.save(p / "test_predictions_v_hat.npy", outputs["v_hat"])
    np.save(p / "test_ground_truth.npy", outputs["truth"])
    np.save(r / "CC_STMT_pred.npy", outputs["expected"])
    np.save(r / "CC_STMT_truth.npy", outputs["truth"])
    metrics = calc_metrics(outputs["expected"], outputs["truth"])
    pd.DataFrame([metrics]).to_csv(r / "cc_stmt_prediction_metrics.csv", index=False)
    with open(r / "cc_stmt_prediction_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
