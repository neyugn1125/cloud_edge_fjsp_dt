import json

with open("data/benchmarks/cloud_baseline_schedule.json") as f:
    ops = json.load(f)["operations_schedule"]

for i in range(25, 45):
    op = ops[i]
    print(f"Op {i+1:2d}: J{op['job_index']}, {op['machine_id']}, start={op.get('start_time', 0):.1f}, end={op.get('end_time', 0):.1f}")
