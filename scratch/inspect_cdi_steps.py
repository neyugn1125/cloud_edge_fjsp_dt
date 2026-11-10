import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from scratch.test_clean_hil import clean_run_closed_loop_hil_simulation

res = clean_run_closed_loop_hil_simulation(seed=201, mode=3, cdi_threshold=0.12)
print("Resync points:", res["resync_points"])
print("Cloud calls:", res["cloud_calls"])

for step_i, cdi_val in enumerate(res["cdi_trace"], 1):
    if step_i in range(18, 25) or step_i in range(35, 42):
        print(f"Step {step_i:2d}: CDI = {cdi_val:.4f}")
