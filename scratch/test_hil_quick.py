import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
from tests.test_task4_3_full_hil import run_closed_loop_hil_simulation
from utils.eval_protocol import TEST_SEEDS, VAL_SEEDS

print("Testing Mode 1, 2, 3 on seed 201...")
for m in [1, 2, 3]:
    res = run_closed_loop_hil_simulation(seed=201, mode=m, cdi_threshold=0.12)
    print(f"Mode {m}: Cmax={res['makespan_min']:.2f}, Cost=${res['tou_cost_usd']:.3f}, Calls={res['cloud_calls']}, PMs={res['num_pm_replacements']}, BDs={res['num_breakdowns']}, MaxCDI={res['max_cdi']:.4f}")
