import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path.cwd()))
from sim_env.fjsp_gym_env import GreenFJSPEnv
from edge_layer.breadboard_daq import BreadboardDAQInterface, RealTimeToolWearEstimator
from tests.test_task4_3_full_hil import (
    compute_planned_trajectories,
    sync_cloud_reference_to_env,
    DEFAULT_ALPHA_CDI,
)
from scratch.test_clean_hil import clean_fast_cloud_reschedule_active

seed = 201
with open("data/benchmarks/cloud_baseline_schedule.json", "r", encoding="utf-8") as f:
    baseline_data = json.load(f)

ops_schedule = baseline_data["operations_schedule"]
num_ops = len(ops_schedule)

env = GreenFJSPEnv(benchmark_path="data/benchmarks/Mk01_green.json", cloud_guided_mask=True)
obs, info = env.reset(seed=seed, options={"realistic_disturbance": True})
current_cloud_ops = list(ops_schedule)
sync_cloud_reference_to_env(env, current_cloud_ops)

cmax_nominal, energy_nominal, planned_cum_cmax, planned_cum_energy = compute_planned_trajectories(
    env, current_cloud_ops
)

daqs = {m_id: BreadboardDAQInterface(machine_id=m_id, mode="hardware_emulator") for m_id in env.machine_ids}
estimators = {m_id: RealTimeToolWearEstimator(machine_id=m_id) for m_id in env.machine_ids}
for m_id in env.machine_ids:
    daqs[m_id].send_command("CMD:CUTTING")
    estimators[m_id].reset(env.machines[m_id].base_initial_wear_um)

cloud_calls = 0
resync_points = []
anchor_step = 0
c_act_anchor = 0.0
e_act_anchor = 0.0
eff_threshold = 0.12

op_shock1 = 20
op_shock2 = 38
halt_min1 = 8.8
halt_min2 = 8.8
wear_spike1 = 26.0
wear_spike2 = 26.0

for step_i in range(1, num_ops + 1):
    op = current_cloud_ops[step_i - 1]
    j = int(op["job_index"])
    m_id = op["machine_id"]
    m_idx = env.machine_ids.index(m_id)
    twin = env.machines[m_id]
    est = estimators[m_id]

    # If disruption occurs at step_i on active machine m_id:
    if step_i in [op_shock1, op_shock2]:
        halt = halt_min1 if step_i == op_shock1 else halt_min2
        wear_sp = wear_spike1 if step_i == op_shock1 else wear_spike2
        op_start = max(twin.available_time, env.job_ready_time[j])
        twin.available_time = op_start + halt
        env.job_ready_time[j] = op_start + halt
        twin.e_stdby_kwh += (halt / 60.0) * twin.p_stdby_kw
        env.total_cost += env._integrate_tou_cost(twin.p_stdby_kw, op_start, op_start + halt)
        twin.current_wear_um = min(99.0, twin.current_wear_um + wear_sp)

    w_hat = est.estimated_wear_um
    sigma = est.sigma_ekf
    t_start = max(twin.available_time, env.job_ready_time[j])
    rate = env.system_config.get_tou_rate(t_start)

    is_peak = (rate >= 0.15)
    s = 1 if is_peak else int(op.get("speed_level", 2))
    alt = next(item for item in env.jobs_data[j]["operations"][env.current_op_idx[j]]["eligible_machines"] if item["machine_id"] == m_id)
    t_base = float(alt["t_base_min"])
    v_s = env.system_config.speed_scaling.levels[s].speed_factor
    delta_w_nom = twin.base_eta * (v_s ** (twin.beta - 1)) * t_base
    w_proj = w_hat + delta_w_nom + 2.0 * sigma
    pm = 1 if ((w_proj >= 82.0 or (w_hat + 2.0 * sigma >= 78.0)) and w_hat >= 50.5) else 0

    act = env.encode_action(j, m_idx, s, pm)
    obs, r, term, tr, info = env.step(act)

    if pm == 1:
        estimators[m_id].reset(0.0)
    daqs[m_id].set_operating_conditions(speed_level=s, true_wear_um=twin.current_wear_um)
    for _ in range(4):
        tel = daqs[m_id].read_telemetry(dt_ms=100)
        estimators[m_id].estimate_wear(tel, speed_level=s, sensor_mode="fusion", use_filter=True)

    k = step_i
    c_act_k = float(info["cmax"])
    e_act_k = float(info.get("total_energy_kwh", 0.0))

    k_idx = min(k, len(planned_cum_cmax) - 1)
    anc_idx = min(anchor_step, len(planned_cum_cmax) - 1)

    c_plan_k = planned_cum_cmax[k_idx]
    c_plan_anc = planned_cum_cmax[anc_idx]
    e_plan_k = planned_cum_energy[k_idx]
    e_plan_anc = planned_cum_energy[anc_idx]

    delta_c = abs((c_act_k - c_act_anchor) - (c_plan_k - c_plan_anc)) / max(1e-6, cmax_nominal)
    delta_e = abs((e_act_k - e_act_anchor) - (e_plan_k - e_plan_anc)) / max(1e-6, energy_nominal)
    cdi = DEFAULT_ALPHA_CDI * delta_c + (1.0 - DEFAULT_ALPHA_CDI) * delta_e

    if step_i in [19, 20, 21, 37, 38, 39]:
        print(f"Step {step_i:2d}: act_m_id={m_id}, c_act_k={c_act_k:.2f}, c_anc={c_act_anchor:.2f}, c_plan_k={c_plan_k:.2f}, c_plan_anc={c_plan_anc:.2f}, delta_c={delta_c:.4f}, cdi={cdi:.4f}")

    if cdi > eff_threshold and step_i < num_ops:
        cloud_calls += 1
        resync_points.append(step_i)
        anchor_step = step_i
        c_act_anchor = c_act_k
        e_act_anchor = e_act_k
        print(f"--> RESYNC TRIGGERED AT STEP {step_i}! cdi={cdi:.4f}")

        current_cloud_ops = clean_fast_cloud_reschedule_active(
            env, current_cloud_ops, estimators=estimators, force_s1=False
        )
        sync_cloud_reference_to_env(env, current_cloud_ops)
        cmax_nominal, energy_nominal, planned_cum_cmax, planned_cum_energy = compute_planned_trajectories(
            env, current_cloud_ops, anchor_step=anchor_step, c_anchor=c_act_anchor, e_anchor=e_act_anchor
        )

print("Final resync points:", resync_points)
print("Final cloud calls:", cloud_calls)
