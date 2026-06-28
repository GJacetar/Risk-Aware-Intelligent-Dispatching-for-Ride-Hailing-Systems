from pathlib import Path
import sys
d = Path(__file__).resolve().parent
sys.path.insert(0, str(d.parent / "common"))
from dispatch_core import CityConfig, run_cli

cfg = CityConfig(
    city="Chicago-B",
    root_dir="/root/autodl-tmp/new",
    proc_dir="/root/autodl-tmp/new/chicago-bigfull/processed",
    raw_dir="/root/autodl-tmp/new/chicago-bigfull/raw",
    out_dir="/root/autodl-tmp/new/chicago-bigfull/dispatch/results",
    model_path="/root/autodl-tmp/new/chicago-bigfull/prediction/models/best_CC_STMT_model.pth",
    base_time="2025-12-01 00:00:00",
    pred_slot_minutes=15,
    city_type="chicago_area",
    default_fleets=(900, 1200, 1500),
    default_max_orders=400000,
    default_max_pickup_dist=3.5
)

if __name__ == "__main__":
    run_cli(cfg)
