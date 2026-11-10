import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from scratch.test_clean_hil import clean_run_closed_loop_hil_simulation
from utils.eval_protocol import TEST_SEEDS, VAL_SEEDS

print("Running 15 TEST_SEEDS for Mode 1, Mode 2, Mode 3...")
m1_calls, m2_calls, m3_calls = [], [], []
m1_cmax, m2_cmax, m3_cmax = [], [], []
m1_cost, m2_cost, m3_cost = [], [], []

for s in TEST_SEEDS:
    r1 = clean_run_closed_loop_hil_simulation(seed=s, mode=1)
    r2 = clean_run_closed_loop_hil_simulation(seed=s, mode=2)
    r3 = clean_run_closed_loop_hil_simulation(seed=s, mode=3, cdi_threshold=0.12)

    m1_calls.append(r1["cloud_calls"])
    m2_calls.append(r2["cloud_calls"])
    m3_calls.append(r3["cloud_calls"])

    m1_cmax.append(r1["makespan_min"])
    m2_cmax.append(r2["makespan_min"])
    m3_cmax.append(r3["makespan_min"])

    m1_cost.append(r1["tou_cost_usd"])
    m2_cost.append(r2["tou_cost_usd"])
    m3_cost.append(r3["tou_cost_usd"])

print(f"Mode 1: Calls={np.mean(m1_calls):.1f}, Cmax={np.mean(m1_cmax):.2f}m, Cost=${np.mean(m1_cost):.3f}")
print(f"Mode 2: Calls={np.mean(m2_calls):.1f}, Cmax={np.mean(m2_cmax):.2f}m, Cost=${np.mean(m2_cost):.3f}")
print(f"Mode 3: Calls={np.mean(m3_calls):.1f}, Cmax={np.mean(m3_cmax):.2f}m, Cost=${np.mean(m3_cost):.3f}")
reduction = (np.mean(m1_calls) - np.mean(m3_calls)) / np.mean(m1_calls) * 100.0
print(f"Cloud Call Reduction: {reduction:.1f}%")
print(f"Makespan Improvement (M2 vs M3): {np.mean(m2_cmax) - np.mean(m3_cmax):.2f} min")
print(f"Cost Improvement (M2 vs M3): ${np.mean(m2_cost) - np.mean(m3_cost):.3f}")
