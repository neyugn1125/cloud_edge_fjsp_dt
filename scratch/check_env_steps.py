import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from sim_env.fjsp_gym_env import GreenFJSPEnv
import json

env = GreenFJSPEnv(benchmark_path="data/benchmarks/Mk01_green.json")
obs, info = env.reset(seed=201)

with open("data/benchmarks/cloud_baseline_schedule.json") as f:
    ops = json.load(f)["operations_schedule"]

for i in range(25):
    op = ops[i]
    j = int(op["job_index"])
    m_id = op["machine_id"]
    m_idx = env.machine_ids.index(m_id)
    s = int(op.get("speed_level", 1))
    pm = 0
    act = env.encode_action(j, m_idx, s, pm)
    obs, r, term, tr, info = env.step(act)
    print(f"Step {i+1:2d} (J{j}, {m_id}): cmax={info['cmax']:.2f}, mach_avails={[round(m.available_time, 1) for m in env.machines.values()]}")
