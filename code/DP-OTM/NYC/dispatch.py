from pathlib import Path
import sys
d = Path(__file__).resolve().parent
sys.path.insert(0, str(d.parent / "common"))
from dispatch_core import CityConfig, run_cli

cfg = CityConfig(
    city="NYC",
    root_dir="/root/autodl-tmp/new",
    proc_dir="/root/autodl-tmp/new/data/processed",
    raw_dir="/root/autodl-tmp/new/data",
    out_dir="/root/autodl-tmp/new/dispatch_results/NYC",
    model_path="/root/autodl-tmp/new/models/NYC/best_CC_STMT_model.pth",
    base_time="2026-01-01 00:00:00",
    pred_slot_minutes=5,
    city_type="nyc",
    default_fleets=(1500, 3000, 6000),
    default_max_orders=150000,
    default_max_pickup_dist=3.0
)

if __name__ == "__main__":
    run_cli(cfg)
