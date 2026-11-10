import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from sim_env.fjsp_gym_env import GreenFJSPEnv
from edge_layer.breadboard_daq import BreadboardDAQInterface, RealTimeToolWearEstimator
from tests.test_task4_3_full_hil import (
    compute_planned_trajectories,
    sync_cloud_reference_to_env,
    DEFAULT_ALPHA_CDI,
)

with open("data/benchmarks/cloud_baseline_schedule.json") as f:
    ops = json.load(f)["operations_schedule"]

env = GreenFJSPEnv(benchmark_path="data/benchmarks/Mk01_green.json")
obs, info = env.reset(seed=201)
cmax_nominal, energy_nominal, planned_cum_cmax, planned_cum_energy = compute_planned_trajectories(env, ops)

halt_min1 = 8.8
wear_spike1 = 26.0

for step_i in range(1, 30):
    op = ops[step_i - 1]
    j = int(op["job_index"])
    m_id = op["machine_id"]
    m_idx = env.machine_ids.index(m_id)

    if step_i == 20:
        m_t = env.machines[m_id]
        m_t.available_time += halt_min1
        env.job_ready_time[j] += halt_min1
        m_t.current_wear_um = min(99.0, m_t.current_wear_um + wear_spike1)

    act = env.encode_action(j, m_idx, int(op.get("speed_level", 1)), 0)
    obs, r, term, tr, info = env.step(act)

    c_act = info["cmax"]
    c_plan = planned_cum_cmax[step_i]
    delta_c = abs(c_act - c_plan) / cmax_nominal
    print(f"Step {step_i:2d}: c_act={c_act:.2f}, c_plan={c_plan:.2f}, delta_c={delta_c:.4f}")
