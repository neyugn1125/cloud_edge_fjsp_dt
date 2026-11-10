import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from scratch.test_clean_hil import clean_run_closed_loop_hil_simulation

res = clean_run_closed_loop_hil_simulation(seed=201, mode=3, cdi_threshold=0.12)
for i, c in enumerate(res["cdi_trace"], 1):
    print(f"Step {i:2d}: CDI = {c:.4f}")
