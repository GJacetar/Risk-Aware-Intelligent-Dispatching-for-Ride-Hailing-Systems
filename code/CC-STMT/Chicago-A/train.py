import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(COMMON_DIR))

from model import CC_STMT
from model import cc_stmt_loss

DEFAULT_PROCESSED_DIR = "/root/autodl-tmp/new/data/chiago/processed"
DEFAULT_MODEL_DIR = "/root/autodl-tmp/new/models/Chicago-A"
DEFAULT_RESULT_DIR = "/root/autodl-tmp/new/results/Chicago-A"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def align_weather(weather, total_slots, num_nodes):
    if weather.ndim == 2:
        weather = np.expand_dims(weather, axis=1).repeat(num_nodes, axis=1)
    elif weather.ndim == 3 and weather.shape[1] == 1:
        weather = np.repeat(weather, num_nodes, axis=1)
    if weather.shape[0] != total_slots:
        n = min(weather.shape[0], total_slots)
        weather = weather[:n]
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
    arrays = {
        "demand": demand,
        "speed": speed,
        "enroute": enroute,
        "weather": weather,
        "adj_spatial": adj_spatial,
        "adj_semantic": adj_semantic
    }
    return crop_arrays(arrays)

def normalize_arrays(arrays, train_end):
    demand = arrays["demand"]
    speed = arrays["speed"]
    enroute = arrays["enroute"]
    weather = arrays["weather"]
    stats = {}
    stats["mean_order"] = np.array(demand[:train_end].mean(), dtype=np.float32)
    stats["std_order"] = np.array(demand[:train_end].std() + 1e-8, dtype=np.float32)
    stats["mean_speed"] = np.array(speed[:train_end].mean(), dtype=np.float32)
    stats["std_speed"] = np.array(speed[:train_end].std() + 1e-8, dtype=np.float32)
    stats["mean_enroute"] = np.array(enroute[:train_end].mean(), dtype=np.float32)
    stats["std_enroute"] = np.array(enroute[:train_end].std() + 1e-8, dtype=np.float32)
    stats["mean_weather"] = weather[:train_end].mean(axis=(0, 1)).astype(np.float32)
    stats["std_weather"] = (weather[:train_end].std(axis=(0, 1)) + 1e-8).astype(np.float32)
    norm = {}
    norm["demand"] = (demand - float(stats["mean_order"])) / float(stats["std_order"])
    norm["speed"] = (speed - float(stats["mean_speed"])) / float(stats["std_speed"])
    norm["enroute"] = (enroute - float(stats["mean_enroute"])) / float(stats["std_enroute"])
    norm["weather"] = (weather - stats["mean_weather"]) / stats["std_weather"]
    return norm, stats

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

def run_epoch(model, loader, adj_spatial, adj_semantic, optimizer, mean_order, std_order, warmup, device):
    model.train()
    total_loss = 0.0
    count = 0
    for xo, xs, xe, xw, yo, ys in loader:
        xo = xo.to(device)
        xs = xs.to(device)
        xe = xe.to(device)
        xw = xw.to(device)
        yo = yo.to(device)
        ys = ys.to(device)
        optimizer.zero_grad(set_to_none=True)
        pi, mu, theta, v_hat = model(xo, xs, xe, xw, adj_spatial, adj_semantic)
        mu = mu * std_order + mean_order
        loss = cc_stmt_loss(pi, mu, theta, v_hat, yo, ys, is_warmup=warmup)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        total_loss += float(loss.item())
        count += 1
    return total_loss / max(count, 1)

def validate(model, loader, adj_spatial, adj_semantic, mean_order, std_order, device):
    model.eval()
    loss_total = 0.0
    mae_total = 0.0
    count = 0
    with torch.no_grad():
        for xo, xs, xe, xw, yo, ys in loader:
            xo = xo.to(device)
            xs = xs.to(device)
            xe = xe.to(device)
            xw = xw.to(device)
            yo = yo.to(device)
            ys = ys.to(device)
            pi, mu, theta, v_hat = model(xo, xs, xe, xw, adj_spatial, adj_semantic)
            mu = mu * std_order + mean_order
            loss = cc_stmt_loss(pi, mu, theta, v_hat, yo, ys, is_warmup=False)
            pred = (1.0 - pi) * mu
            loss_total += float(loss.item())
            mae_total += float(torch.mean(torch.abs(pred - yo)).item())
            count += 1
    return loss_total / max(count, 1), mae_total / max(count, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.8)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--fine_tune_epochs", type=int, default=80)
    parser.add_argument("--lr_warmup", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    args = parser.parse_args()
    set_seed(args.seed)
    ensure_dir(args.model_dir)
    ensure_dir(args.result_dir)
    arrays = load_arrays(args.processed_dir)
    total_slots = arrays["demand"].shape[0]
    num_nodes = arrays["demand"].shape[1]
    train_end = int(total_slots * args.train_ratio)
    val_end = int(total_slots * args.val_ratio)
    norm, stats = normalize_arrays(arrays, train_end)
    stats["train_end"] = np.array(train_end, dtype=np.int64)
    stats["val_end"] = np.array(val_end, dtype=np.int64)
    stats["total_slots"] = np.array(total_slots, dtype=np.int64)
    stats["num_nodes"] = np.array(num_nodes, dtype=np.int64)
    stats["seq_len"] = np.array(args.seq_len, dtype=np.int64)
    stats["weather_dim"] = np.array(arrays["weather"].shape[-1], dtype=np.int64)
    np.savez(Path(args.model_dir) / "norm_stats.npz", **stats)
    train_dataset = RideHailingDataset(norm, arrays["demand"], arrays["speed"], 0, train_end, args.seq_len)
    val_dataset = RideHailingDataset(norm, arrays["demand"], arrays["speed"], train_end, val_end, args.seq_len)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("The training or validation dataset is empty")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 3 + arrays["weather"].shape[-1]
    model = CC_STMT(num_nodes=num_nodes, input_dim=input_dim, hidden_dim=64, num_layers=2, nhead=8, seq_len=args.seq_len).to(device)
    adj_spatial = torch.FloatTensor(arrays["adj_spatial"]).to(device)
    adj_semantic = torch.FloatTensor(arrays["adj_semantic"]).to(device)
    mean_order = float(stats["mean_order"])
    std_order = float(stats["std_order"])
    optimizer = optim.AdamW(model.parameters(), lr=args.lr_warmup, weight_decay=args.weight_decay)
    history = []
    best_mae = float("inf")
    no_improve = 0
    for epoch in range(args.warmup_epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, adj_spatial, adj_semantic, optimizer, mean_order, std_order, True, device)
        val_loss, val_mae = validate(model, val_loader, adj_spatial, adj_semantic, mean_order, std_order, device)
        history.append({"phase": "warmup", "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_mae": val_mae, "time_sec": time.time() - t0})
        print(f"warmup {epoch + 1}/{args.warmup_epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_mae={val_mae:.6f}", flush=True)
    for group in optimizer.param_groups:
        group["lr"] = args.lr_finetune
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    for epoch in range(args.fine_tune_epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, adj_spatial, adj_semantic, optimizer, mean_order, std_order, False, device)
        val_loss, val_mae = validate(model, val_loader, adj_spatial, adj_semantic, mean_order, std_order, device)
        scheduler.step(val_mae)
        history.append({"phase": "finetune", "epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_mae": val_mae, "time_sec": time.time() - t0})
        print(f"finetune {epoch + 1}/{args.fine_tune_epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_mae={val_mae:.6f}", flush=True)
        if val_mae < best_mae:
            best_mae = val_mae
            no_improve = 0
            torch.save(model.state_dict(), Path(args.model_dir) / "best_cc_stmt_model.pth")
        else:
            no_improve += 1
            if no_improve >= args.early_stop_patience:
                break
    pd.DataFrame(history).to_csv(Path(args.result_dir) / "cc_stmt_train_log.csv", index=False)
    with open(Path(args.result_dir) / "cc_stmt_train_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_mae": best_mae, "total_slots": total_slots, "num_nodes": num_nodes}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
