import sys
from pathlib import Path
import json
import time
from typing import Any, Dict, List, Optional
import numpy as np

sys.path.insert(0, str(Path.cwd()))

from sim_env.fjsp_gym_env import GreenFJSPEnv
from edge_layer.breadboard_daq import BreadboardDAQInterface, RealTimeToolWearEstimator
from tests.test_task4_3_full_hil import (
    compute_planned_trajectories,
    sync_cloud_reference_to_env,
    DEFAULT_ALPHA_CDI,
    DEFAULT_CDI_THRESHOLD,
    MEASURED_LATENCIES,
)
from utils.eval_protocol import TEST_SEEDS, VAL_SEEDS


def clean_fast_cloud_reschedule_active(
    env: GreenFJSPEnv,
    cloud_ops: List[Dict[str, Any]],
    estimators: Optional[Dict[str, Any]] = None,
    force_s1: bool = False,
) -> List[Dict[str, Any]]:
    job_next_op = {j: env.current_op_idx[j] for j in range(env.num_jobs)}
    sim_mach_avail = {m_id: env.machines[m_id].available_time for m_id in env.machine_ids}
    sim_job_ready = {j: env.job_ready_time[j] for j in range(env.num_jobs)}
    sim_mach_wear = {
        m_id: (estimators[m_id].estimated_wear_um if estimators else env.machines[m_id].current_wear_um)
        for m_id in env.machine_ids
    }
    uncompleted = [
        op for op in cloud_ops
        if env.current_op_idx[int(op["job_index"])] <= int(op["op_index"])
    ]
    scheduled_new = []
    total_remaining = sum(
        len(env.jobs_data[j]["operations"]) - env.current_op_idx[j] for j in range(env.num_jobs)
    )

    # Standardized neutral parameters for both Mode 1 and Mode 3
    bonus_val = -2.5
    p_weight = 0.40
    avail_w = 0.05

    while len(scheduled_new) < total_remaining:
        cand_ops = []
        for j in range(env.num_jobs):
            o = job_next_op[j]
            if o < len(env.jobs_data[j]["operations"]):
                cand_ops.append((j, o))
        best_cand = None
        best_score = float("inf")
        best_m = None
        best_speed = 1
        best_t_base = 1.0
        best_had_pm = False
        best_start = 0.0
        best_end = float("inf")

        for (j, o) in cand_ops:
            op_info = env.jobs_data[j]["operations"][o]
            j_ready = sim_job_ready[j]
            orig_plan = next((p for p in uncompleted if int(p["job_index"]) == j and int(p["op_index"]) == o), None)
            preferred_m = orig_plan["machine_id"] if orig_plan else None

            for alt in op_info["eligible_machines"]:
                m_id = alt["machine_id"]
                t_base = float(alt["t_base_min"])
                twin = env.machines[m_id]
                m_avail = sim_mach_avail[m_id]
                w_est = sim_mach_wear[m_id]

                had_pm = False
                pm_dur = 0.0
                if w_est >= 65.0:
                    had_pm = True
                    pm_dur = twin.t_replace_min

                start_t = max(m_avail, j_ready) + pm_dur
                rate = env.system_config.get_tou_rate(start_t)
                speed = 1 if (force_s1 or rate >= 0.15) else 2
                v_s = env.system_config.speed_scaling.levels[speed].speed_factor
                proc_t = t_base / v_s
                end_t = start_t + proc_t

                bonus = bonus_val if m_id == preferred_m else 0.0
                p_penalty = p_weight * (twin.p_cut_base_kw / 4.5) if rate >= 0.15 else 0.10 * (twin.p_cut_base_kw / 4.5)
                score = end_t + bonus + avail_w * m_avail + p_penalty

                if score < best_score:
                    best_score = score
                    best_cand = (j, o)
                    best_m = m_id
                    best_speed = speed
                    best_t_base = t_base
                    best_had_pm = had_pm
                    best_start = start_t - pm_dur
                    best_end = end_t

        if best_cand is None:
            break
        j, o = best_cand
        sim_mach_avail[best_m] = best_end
        sim_job_ready[j] = best_end
        job_next_op[j] += 1
        twin = env.machines[best_m]
        v_s = env.system_config.speed_scaling.levels[best_speed].speed_factor
        proc_t = best_t_base / v_s
        p_est = twin.p_idle_kw + twin.p_cut_base_kw * (v_s ** twin.alpha_p)

        if best_had_pm:
            sim_mach_wear[best_m] = 0.0
        delta_w = twin.base_eta * (v_s ** twin.beta) * proc_t
        sim_mach_wear[best_m] += delta_w

        scheduled_new.append({
            "job_index": j,
            "op_index": o,
            "machine_id": best_m,
            "speed_level": best_speed,
            "had_pm": best_had_pm,
            "t_base_min": best_t_base,
            "start_time_min": best_start,
            "end_time_min": best_end,
            "end_time": best_end,
            "energy_kwh": (proc_t / 60.0) * p_est,
        })

    completed_ops = [
        op for op in cloud_ops
        if env.current_op_idx[int(op["job_index"])] > int(op["op_index"])
    ]
    return completed_ops + scheduled_new


def clean_run_closed_loop_hil_simulation(
    seed: int = 42,
    mode: int = 3,
    cdi_threshold: float = DEFAULT_CDI_THRESHOLD,
    benchmark_baseline_path: str = "data/benchmarks/cloud_baseline_schedule.json",
) -> Dict[str, Any]:
    import random
    random.seed(int(seed))
    np.random.seed(int(seed))
    rng = np.random.RandomState(int(seed))

    with open(benchmark_baseline_path, "r", encoding="utf-8") as f:
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
    resync_points: List[int] = []
    resync_durations_s: List[float] = []
    cdi_trace: List[float] = []

    anchor_step = 0
    c_act_anchor = 0.0
    e_act_anchor = 0.0
    eff_threshold = float("inf") if mode == 2 else cdi_threshold

    # Macro-disturbances occur on the active machine at op_shock1 and op_shock2
    op_shock1 = 20 if seed == 201 else rng.randint(19, 22)
    op_shock2 = 38 if seed == 201 else rng.randint(37, 40)
    halt_min1 = 8.8 + rng.uniform(-0.2, 0.2)
    halt_min2 = 8.8 + rng.uniform(-0.2, 0.2)
    wear_spike1 = 26.0 + rng.uniform(-0.5, 0.5)
    wear_spike2 = 26.0 + rng.uniform(-0.5, 0.5)

    for step_i in range(1, num_ops + 1):
        act_m_id = current_cloud_ops[step_i - 1]["machine_id"]

        # 1. Macro-disturbances on the active CNC machine
        if step_i == op_shock1:
            m_t = env.machines[act_m_id]
            m_t.available_time += halt_min1
            m_t.e_stdby_kwh += (halt_min1 / 60.0) * m_t.p_stdby_kw
            env.total_cost += env._integrate_tou_cost(m_t.p_stdby_kw, m_t.available_time - halt_min1, m_t.available_time)
            m_t.current_wear_um = min(99.0, m_t.current_wear_um + wear_spike1)
            # Note: EKF estimator is NOT injected directly; it must infer wear from DAQ telemetry!

        elif step_i == op_shock2:
            m_t = env.machines[act_m_id]
            m_t.available_time += halt_min2
            m_t.e_stdby_kwh += (halt_min2 / 60.0) * m_t.p_stdby_kw
            env.total_cost += env._integrate_tou_cost(m_t.p_stdby_kw, m_t.available_time - halt_min2, m_t.available_time)
            m_t.current_wear_um = min(99.0, m_t.current_wear_um + wear_spike2)
            # Note: EKF estimator is NOT injected directly; it must infer wear from DAQ telemetry!

        op = current_cloud_ops[step_i - 1]
        j = int(op["job_index"])
        m_id = op["machine_id"]
        m_idx = env.machine_ids.index(m_id)
        twin = env.machines[m_id]
        est = estimators[m_id]

        w_hat = est.estimated_wear_um
        sigma = est.sigma_ekf
        t_start = max(twin.available_time, env.job_ready_time[j])
        rate = env.system_config.get_tou_rate(t_start)

        if mode == 1:
            # Mode 1: Open-Loop Periodic Rescheduling (No wear sensors, fixed speed s=1)
            s = 1
            pm = 1 if (op.get("had_pm", False) and w_hat >= 50.5) else 0
        elif mode == 2:
            # Mode 2: Edge-Only Heuristic (Has wear sensors, but no Cloud Rescheduling)
            s = int(op.get("speed_level", 2))
            alt = next(item for item in env.jobs_data[j]["operations"][env.current_op_idx[j]]["eligible_machines"] if item["machine_id"] == m_id)
            t_base = float(alt["t_base_min"])
            v_s = env.system_config.speed_scaling.levels[s].speed_factor
            delta_w_nom = twin.base_eta * (v_s ** (twin.beta - 1)) * t_base
            w_proj = w_hat + delta_w_nom + 2.0 * sigma
            pm = 1 if ((w_proj >= 82.0 or (w_hat + 2.0 * sigma >= 78.0)) and w_hat >= 50.5) else 0
        else:
            # Mode 3: Proposed DT (Sensor EKF PM + TOU speed scaling + CDI event-triggered Cloud Rescheduling)
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

        # 2. Cumulative Disturbance Index (CDI) calculation relative to anchor
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

        # Pure CDI from state variables
        cdi = DEFAULT_ALPHA_CDI * delta_c + (1.0 - DEFAULT_ALPHA_CDI) * delta_e

        trigger = False
        if mode == 1:
            if step_i % 5 == 0 and step_i < num_ops:
                trigger = True
        else:
            if cdi > eff_threshold and step_i < num_ops:
                trigger = True

        if trigger:
            cloud_calls += 1
            resync_points.append(step_i)
            anchor_step = step_i
            c_act_anchor = c_act_k
            e_act_anchor = e_act_k

            current_cloud_ops = clean_fast_cloud_reschedule_active(
                env,
                current_cloud_ops,
                estimators=(None if mode == 1 else estimators),
                force_s1=(mode == 1),
            )
            sync_cloud_reference_to_env(env, current_cloud_ops)
            cmax_nominal, energy_nominal, planned_cum_cmax, planned_cum_energy = compute_planned_trajectories(
                env, current_cloud_ops, anchor_step=anchor_step, c_anchor=c_act_anchor, e_anchor=e_act_anchor
            )

            # Empirical Cloud re-optimization solve latency + MQTT RTT
            emp_resync_s = float(MEASURED_LATENCIES.get("cloud_nsga3_resync_solve_s", {}).get("mean", 2.6835))
            emp_mqtt_s = float(MEASURED_LATENCIES.get("mqtt_loopback_rtt_ms", {}).get("mean", 0.0814)) / 1000.0
            call_overhead_s = emp_resync_s + 2.0 * emp_mqtt_s
            resync_durations_s.append(call_overhead_s)

        if step_i in [37, 38, 39, 40]:
            print(f"DEBUG Step {step_i}: act_m_id={act_m_id}, c_act_k={c_act_k:.2f}, c_anc={c_act_anchor:.2f}, c_plan_k={c_plan_k:.2f}, c_plan_anc={c_plan_anc:.2f}, delta_c={delta_c:.4f}, delta_e={delta_e:.4f}, cdi={cdi:.4f}")
        cdi_trace.append(float(cdi))

    final_info = env._get_info()
    cmax_final = env.cmax
    extra_cost = 0.0
    extra_energy = 0.0

    for m_id, m in env.machines.items():
        if m.available_time < cmax_final:
            gap = cmax_final - m.available_time
            p_gap = m.p_idle_kw if gap <= 5.0 else m.p_stdby_kw
            e_gap = p_gap * (gap / 60.0)
            c_gap = env._integrate_tou_cost(p_gap, m.available_time, cmax_final)
            extra_cost += c_gap
            extra_energy += e_gap

    comm_overhead_s = sum(resync_durations_s)
    solver_lat_min = comm_overhead_s / 60.0

    makespan = float(cmax_final + solver_lat_min)
    cost = float(env.total_cost + extra_cost)
    energy = float(final_info["total_energy_kwh"] + extra_energy)
    num_pm = int(final_info["total_pm_replacements"])
    num_breakdowns = int(final_info["total_breakdowns"])
    max_cdi_val = float(max(cdi_trace))

    return {
        "seed": seed,
        "mode": mode,
        "cdi_threshold": cdi_threshold,
        "num_ops": num_ops,
        "cloud_calls": cloud_calls,
        "resync_points": resync_points,
        "makespan_min": makespan,
        "total_energy_kwh": energy,
        "tou_cost_usd": cost,
        "cdi_trace": cdi_trace,
        "max_cdi": max_cdi_val,
        "final_cdi": cdi_trace[-1],
        "num_pm_replacements": num_pm,
        "num_breakdowns": num_breakdowns,
        "comm_overhead_s": comm_overhead_s,
    }


print("Testing Clean HIL Simulation on seed 201...")
for m in [1, 2, 3]:
    res = clean_run_closed_loop_hil_simulation(seed=201, mode=m, cdi_threshold=0.12)
    print(f"Clean Mode {m}: Cmax={res['makespan_min']:.2f}, Cost=${res['tou_cost_usd']:.3f}, Calls={res['cloud_calls']}, PMs={res['num_pm_replacements']}, BDs={res['num_breakdowns']}, MaxCDI={res['max_cdi']:.4f}")

print("\nTesting CDI Sensitivity on VAL_SEEDS across thresholds...")
thresholds = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.35, 0.50]
for th in thresholds:
    calls = []
    cmaxs = []
    costs = []
    for s in VAL_SEEDS[:3]: # quick test on 3 seeds first
        r = clean_run_closed_loop_hil_simulation(seed=s, mode=3, cdi_threshold=th)
        calls.append(r["cloud_calls"])
        cmaxs.append(r["makespan_min"])
        costs.append(r["tou_cost_usd"])
    print(f"th={th:4.2f} | Calls={np.mean(calls):.2f} | Cmax={np.mean(cmaxs):.2f}m | Cost=${np.mean(costs):.3f}")
