from pathlib import Path
import sys
d = Path(__file__).resolve().parent
sys.path.insert(0, str(d.parent / "common"))
from dispatch_core import CityConfig, run_cli

cfg = CityConfig(
    city="Chicago-A",
    root_dir="/root/autodl-tmp/new",
    proc_dir="/root/autodl-tmp/new/data/chiago/processed",
    raw_dir="/root/autodl-tmp/new/data/chiago",
    out_dir="/root/autodl-tmp/new/dispatch_results/Chicago-A",
    model_path="/root/autodl-tmp/new/models/chiago/best_CC_STMT_model.pth",
    base_time="2025-12-01 00:00:00",
    pred_slot_minutes=15,
    city_type="chicago_tract",
    default_fleets=(450, 550, 650),
    default_max_orders=150000,
    default_max_pickup_dist=3.0
)

if __name__ == "__main__":
    run_cli(cfg)
