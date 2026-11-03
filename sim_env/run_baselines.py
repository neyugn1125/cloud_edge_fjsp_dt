"""
Evaluation of Baseline Dispatching Rules for Green-FJSP Benchmark Mk01.

Implements and benchmarks 4 heuristic dispatching rules on GreenFJSPEnv:
    - Rule 1 (SPT + Fast Speed): Shortest Processing Time, speed s=3 (v=1.5), reactive maintenance.
    - Rule 2 (EET + Eco Speed): Lowest Estimated Energy, speed s=1 (v=1.0), proactive PM @ 85% w_max.
    - Rule 3 (EDD + Balanced Speed): Earliest Due Date, speed s=2 (v=1.25, throttled to s=1 in Peak hours), PM @ 80% w_max.
    - Rule 4 (Random Valid Policy): Uniform random selection among masked valid actions (averaged over 10 seeds).

Generates:
    - Terminal comparative summary table
    - CSV export: data/benchmarks/phase1_baseline_results.csv
    - Dual-panel Gantt and Instantaneous Power Profile: tests/phase1_sample_gantt.png
"""

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from sim_env.fjsp_gym_env import GreenFJSPEnv


# ==============================================================================
# Dispatching Heuristics
# ==============================================================================

def rule_spt_fast(env: GreenFJSPEnv, mask: np.ndarray) -> int:
    """
    Rule 1 (SPT + Fast Speed):
    Prioritize operation and machine pairing with the shortest processing time.
    Always select maximum cutting speed s=3 (v=1.5).
    Tool replacement only occurs reactively upon catastrophic breakdown (pm_flag=0).
    """
    valid_actions = np.where(mask)[0]
    best_action = None
    best_score = (float("inf"), float("inf"), float("inf"))

    for a in valid_actions:
        j_idx, m_idx, speed_level, pm_flag = env.decode_action(a)
        # Filter for speed=3 and no proactive PM
        if speed_level != 3 or pm_flag != 0:
            continue

        curr_op = env.current_op_idx[j_idx]
        op_info = env.jobs_data[j_idx]["operations"][curr_op]
        m_id = env.machine_ids[m_idx]
        alt = next(a_item for a_item in op_info["eligible_machines"] if a_item["machine_id"] == m_id)

        t_proc = float(alt["t_base_min"]) / 1.5
        m_twin = env.machines[m_id]
        start_t = max(env.job_ready_time[j_idx], m_twin.available_time)

        # Primary key: t_proc; Secondary key: start_time; Tie-breaker: job_idx
        score = (t_proc, start_t, j_idx)
        if score < best_score:
            best_score = score
            best_action = a

    # Defensive fallback
    return best_action if best_action is not None else int(valid_actions[0])


def rule_eet_eco(env: GreenFJSPEnv, mask: np.ndarray) -> int:
    """
    Rule 2 (EET + Eco Speed):
    Prioritize machine and operation pairing with lowest estimated energy consumption.
    Always select economy speed s=1 (v=1.0).
    Trigger proactive PM whenever flank wear reaches or exceeds 85% w_max.
    """
    valid_actions = np.where(mask)[0]
    best_action = None
    best_score = (float("inf"), float("inf"), float("inf"))

    for a in valid_actions:
        j_idx, m_idx, speed_level, pm_flag = env.decode_action(a)
        if speed_level != 1:
            continue

        m_id = env.machine_ids[m_idx]
        m_twin = env.machines[m_id]
        target_pm = 1 if m_twin.current_wear_um >= (0.85 * m_twin.w_max_um) else 0
        if pm_flag != target_pm:
            continue

        curr_op = env.current_op_idx[j_idx]
        op_info = env.jobs_data[j_idx]["operations"][curr_op]
        alt = next(a_item for a_item in op_info["eligible_machines"] if a_item["machine_id"] == m_id)

        t_base = float(alt["t_base_min"])
        wear_ratio = m_twin.current_wear_um / m_twin.w_max_um
        p_est = m_twin.p_idle_kw + m_twin.p_cut_base_kw * (1.0 + m_twin.gamma * wear_ratio)
        e_proc_est = p_est * (t_base / 60.0)

        # Idle gap energy
        start_t = max(env.job_ready_time[j_idx], m_twin.available_time)
        e_gap_est = 0.0
        if start_t > m_twin.available_time:
            gap = start_t - m_twin.available_time
            _, e_gap_est = m_twin.evaluate_idle_gap(gap)

        e_pm_est = m_twin.p_stdby_kw * (m_twin.t_replace_min / 60.0) if target_pm == 1 else 0.0
        total_e_est = e_proc_est + e_gap_est + e_pm_est

        score = (total_e_est, start_t, j_idx)
        if score < best_score:
            best_score = score
            best_action = a

    return best_action if best_action is not None else int(valid_actions[0])


def rule_edd_balanced(env: GreenFJSPEnv, mask: np.ndarray) -> int:
    """
    Rule 3 (EDD + Balanced Speed):
    Prioritize Job with the earliest Due Date (D_i).
    Select balanced speed s=2 (v=1.25), but throttle to s=1 (v=1.0) during On-Peak TOU tariff periods.
    Trigger proactive PM whenever flank wear reaches or exceeds 80% w_max.
    """
    valid_actions = np.where(mask)[0]

    # Check if shop floor is currently in an On-Peak pricing window
    curr_t = min(m.available_time for m in env.machines.values())
    is_peak = env.system_config.get_tou_rate(curr_t) >= 0.20
    target_s = 1 if is_peak else 2

    # Sort unfinished jobs by Due Date ascending
    unfinished_jobs = [
        j for j in range(env.num_jobs)
        if env.current_op_idx[j] < len(env.jobs_data[j]["operations"])
    ]
    unfinished_jobs.sort(key=lambda j: float(env.jobs_data[j]["due_date_min"]))

    for j in unfinished_jobs:
        candidates = []
        for a in valid_actions:
            j_a, m_a, s_a, pm_a = env.decode_action(a)
            if j_a != j:
                continue
            if s_a != target_s:
                continue

            m_id = env.machine_ids[m_a]
            m_twin = env.machines[m_id]
            target_pm = 1 if m_twin.current_wear_um >= (0.80 * m_twin.w_max_um) else 0
            if pm_a != target_pm:
                continue

            start_t = max(env.job_ready_time[j], m_twin.available_time)
            candidates.append((start_t, a))

        if candidates:
            # Select machine providing earliest start time
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

    # Fallback to any valid action matching target speed
    speed_matches = [
        a for a in valid_actions
        if env.decode_action(a)[2] == target_s and env.decode_action(a)[3] == 0
    ]
    if speed_matches:
        return speed_matches[0]

    return int(valid_actions[0])


# ==============================================================================
# Simulation Execution Helpers
# ==============================================================================

def execute_deterministic_rule(
    env: GreenFJSPEnv,
    rule_fn: Callable[[GreenFJSPEnv, np.ndarray], int],
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute a single episode using a deterministic dispatching rule."""
    obs, info = env.reset(seed=seed)
    terminated = False

    while not terminated:
        mask = env.action_masks()
        action = rule_fn(env, mask)
        obs, reward, terminated, truncated, info = env.step(action)

    e_proc = sum(m.e_proc_kwh for m in env.machines.values())
    e_idle_stdby = sum(m.e_idle_kwh + m.e_stdby_kwh for m in env.machines.values())
    e_on_off = sum(m.e_on_off_kwh for m in env.machines.values())
    e_total = e_proc + e_idle_stdby + e_on_off

    return {
        "cmax": info["cmax"],
        "e_total": e_total,
        "e_proc": e_proc,
        "e_idle_stdby": e_idle_stdby,
        "e_on_off": e_on_off,
        "cost": info["total_cost"],
        "twt": info["total_weighted_tardiness"],
        "pms": info["total_pm_replacements"],
        "bds": info["total_breakdowns"],
        "scheduled_events": list(env.scheduled_events),
    }


def execute_random_baseline(
    env: GreenFJSPEnv,
    num_seeds: int = 10,
) -> Dict[str, Any]:
    """Execute random valid policy over multiple seeds and aggregate mean metrics."""
    cmax_list, etot_list, eproc_list, eidle_list, eonoff_list = [], [], [], [], []
    cost_list, twt_list, pm_list, bd_list = [], [], [], []

    for s in range(num_seeds):
        obs, info = env.reset(seed=s)
        rng = np.random.default_rng(s)
        terminated = False

        while not terminated:
            mask = env.action_masks()
            valid_actions = np.where(mask)[0]
            action = int(rng.choice(valid_actions))
            obs, reward, terminated, truncated, info = env.step(action)

        e_proc = sum(m.e_proc_kwh for m in env.machines.values())
        e_idle_stdby = sum(m.e_idle_kwh + m.e_stdby_kwh for m in env.machines.values())
        e_on_off = sum(m.e_on_off_kwh for m in env.machines.values())
        e_total = e_proc + e_idle_stdby + e_on_off

        cmax_list.append(info["cmax"])
        etot_list.append(e_total)
        eproc_list.append(e_proc)
        eidle_list.append(e_idle_stdby)
        eonoff_list.append(e_on_off)
        cost_list.append(info["total_cost"])
        twt_list.append(info["total_weighted_tardiness"])
        pm_list.append(info["total_pm_replacements"])
        bd_list.append(info["total_breakdowns"])

    return {
        "cmax": float(np.mean(cmax_list)),
        "cmax_std": float(np.std(cmax_list)),
        "e_total": float(np.mean(etot_list)),
        "e_proc": float(np.mean(eproc_list)),
        "e_idle_stdby": float(np.mean(eidle_list)),
        "e_on_off": float(np.mean(eonoff_list)),
        "cost": float(np.mean(cost_list)),
        "twt": float(np.mean(twt_list)),
        "pms": float(np.mean(pm_list)),
        "bds": float(np.mean(bd_list)),
    }


# ==============================================================================
# Visualization: Dual-Panel Gantt Chart & Power Profile
# ==============================================================================

def plot_gantt_and_power_profile(
    scheduled_events: List[Dict[str, Any]],
    machines: Dict[str, Any],
    cmax: float,
    p_limit_kw: float,
    tou_tiers: List[Any],
    output_path: str = "tests/phase1_sample_gantt.png",
) -> None:
    """
    Generate and save a publication-quality dual-panel figure:
        - Top panel: Machine timeline Gantt chart colored by Job with Standby & PM annotations.
        - Bottom panel: Total shop-floor instantaneous power demand (kW) with peak ceiling & TOU zones.
    """
    fig, (ax_gantt, ax_power) = plt.subplots(
        2, 1, figsize=(14, 9), dpi=300, sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )

    machine_names = ["M1", "M2", "M3", "M4", "M5", "M6"]
    y_pos = {m_id: idx for idx, m_id in enumerate(machine_names)}

    # Distinct palette for 10 Jobs
    job_cmap = plt.get_cmap("tab10")
    job_colors = {f"J{j+1}": job_cmap(j) for j in range(10)}

    # 1. Plot Gantt Bars
    for ev in scheduled_events:
        m_id = ev["machine_id"]
        y = y_pos[m_id]

        # Idle Gap Standby block
        if ev["idle_action"] in ("STANDBY", "TURN_OFF") and ev["idle_gap_min"] > 0:
            gap_start = ev["mach_ready_before"]
            gap_dur = ev["idle_gap_min"]
            ax_gantt.barh(
                y, gap_dur, left=gap_start, height=0.55,
                color="#e0e0e0", edgecolor="#9e9e9e", hatch="//", alpha=0.8
            )

        # Opportunistic PM block
        if ev["had_pm"]:
            pm_start = ev["t_pm_start"]
            pm_dur = ev["t_pm_end"] - pm_start
            ax_gantt.barh(
                y, pm_dur, left=pm_start, height=0.55,
                color="#ffb74d", edgecolor="#e65100", hatch="xx", alpha=0.95
            )
            ax_gantt.text(
                pm_start + pm_dur / 2, y, "PM",
                ha="center", va="center", fontsize=7.5, fontweight="bold", color="#b71c1c"
            )

        # Cutting Operation block
        cut_start = ev["t_cut_start"]
        cut_dur = ev["t_cut_end"] - cut_start
        j_id = ev["job_id"]
        color = job_colors.get(j_id, "#42a5f5")

        ax_gantt.barh(
            y, cut_dur, left=cut_start, height=0.60,
            color=color, edgecolor="black", linewidth=0.8, alpha=0.9
        )

        label_txt = f"{ev['op_id']}\ns={ev['speed_level']}"
        if cut_dur >= 2.0:
            ax_gantt.text(
                cut_start + cut_dur / 2, y, label_txt,
                ha="center", va="center", fontsize=6.8, fontweight="bold", color="white"
            )
        elif cut_dur >= 1.0:
            ax_gantt.text(
                cut_start + cut_dur / 2, y, ev["op_id"],
                ha="center", va="center", fontsize=6.0, color="white"
            )

        # Breakdown block (if any)
        if ev["had_breakdown"]:
            bd_start = ev["t_bd_start"]
            bd_dur = ev["t_bd_end"] - bd_start
            ax_gantt.barh(
                y, bd_dur, left=bd_start, height=0.55,
                color="#d32f2f", edgecolor="black", hatch="\\\\", alpha=0.95
            )
            ax_gantt.text(
                bd_start + bd_dur / 2, y, "FAIL",
                ha="center", va="center", fontsize=7.5, fontweight="bold", color="white"
            )

    ax_gantt.set_yticks(range(len(machine_names)))
    ax_gantt.set_yticklabels(machine_names, fontsize=10, fontweight="bold")
    ax_gantt.invert_yaxis()  # M1 at the top
    ax_gantt.set_ylabel("CNC Machines", fontsize=11, fontweight="bold")
    ax_gantt.set_title(
        f"Green-FJSP Schedule (Rule 3: EDD + Balanced Speed) | Makespan: {cmax:.2f} min",
        fontsize=12, fontweight="bold"
    )
    ax_gantt.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Job Legend
    job_patches = [
        mpatches.Patch(color=job_colors[f"J{j+1}"], label=f"J{j+1}")
        for j in range(10)
    ]
    stby_patch = mpatches.Patch(facecolor="#e0e0e0", edgecolor="#9e9e9e", hatch="//", label="Standby/Off")
    pm_patch = mpatches.Patch(facecolor="#ffb74d", edgecolor="#e65100", hatch="xx", label="PM (5m)")
    all_legends = job_patches + [stby_patch, pm_patch]
    ax_gantt.legend(
        handles=all_legends, bbox_to_anchor=(1.01, 1.0), loc="upper left",
        fontsize=8.5, framealpha=0.9, title="Jobs & Modes"
    )

    # 2. Reconstruct Total Shop Instantaneous Power Curve
    max_t = max(cmax, 115.0)
    time_grid = np.linspace(0.0, max_t, int(max_t * 10) + 1)
    shop_power = np.zeros_like(time_grid)

    for idx, t in enumerate(time_grid):
        p_t = 0.0
        for ev in scheduled_events:
            # Active cutting
            if ev["t_cut_start"] <= t < ev["t_cut_end"]:
                p_t += ev["p_actual_kw"]
            # PM active
            elif ev["had_pm"] and ev["t_pm_start"] <= t < ev["t_pm_end"]:
                p_t += 0.20  # standby power
            # Breakdown active
            elif ev["had_breakdown"] and ev["t_bd_start"] <= t < ev["t_bd_end"]:
                p_t += 0.20
            # Idle gap standby
            elif ev["idle_action"] == "STANDBY" and ev["mach_ready_before"] <= t < ev["start_time"]:
                p_t += 0.20
            elif ev["idle_action"] == "IDLE" and ev["mach_ready_before"] <= t < ev["start_time"]:
                p_t += 1.10
        shop_power[idx] = p_t

    # TOU tariff background shading
    tou_zones = [
        (0.0, 25.0, "#c8e6c9", "Off-Peak ($0.06/kWh)"),
        (25.0, 60.0, "#ffcdd2", "On-Peak ($0.20/kWh)"),
        (60.0, 100.0, "#fff9c4", "Mid-Peak ($0.12/kWh)"),
        (100.0, max_t, "#ffcdd2", "On-Peak ($0.20/kWh)"),
    ]

    for z_start, z_end, z_col, z_lbl in tou_zones:
        if z_start < max_t:
            z_right = min(z_end, max_t)
            ax_power.axvspan(z_start, z_right, color=z_col, alpha=0.35, zorder=0)
            ax_gantt.axvline(x=z_start, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
            mid_z = (z_start + z_right) / 2
            ax_power.text(
                mid_z, p_limit_kw * 1.08, z_lbl,
                ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#37474f"
            )

    # Bottom Panel: Power Curve
    ax_power.plot(time_grid, shop_power, color="#0d47a1", linewidth=1.8, label="Total Shop Power $P_{total}(t)$", zorder=3)
    ax_power.fill_between(time_grid, 0, shop_power, color="#90caf9", alpha=0.55, zorder=2)

    # Peak Power Limit Ceiling
    ax_power.axhline(
        y=p_limit_kw, color="#b71c1c", linestyle="--", linewidth=1.6,
        label=f"Peak Ceiling $P_{{limit}} = {p_limit_kw:.1f}$ kW", zorder=4
    )

    ax_power.set_xlabel("Simulation Time (Minutes)", fontsize=11, fontweight="bold")
    ax_power.set_ylabel("Shop Power (kW)", fontsize=11, fontweight="bold")
    ax_power.set_xlim(0, max_t)
    ax_power.set_ylim(0, max(np.max(shop_power) * 1.25, p_limit_kw * 1.2))
    ax_power.grid(True, linestyle="--", alpha=0.5)
    ax_power.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, dpi=300)
    plt.close()


# ==============================================================================
# Main Runner & Benchmark Orchestrator
# ==============================================================================

def run_all_baselines(
    benchmark_path: str = "data/benchmarks/Mk01_green.json",
    config_path: str = "config/system_params.yaml",
    csv_output_path: str = "data/benchmarks/phase1_baseline_results.csv",
    gantt_output_path: str = "tests/phase1_sample_gantt.png",
) -> List[Dict[str, Any]]:
    """
    Run all 4 baseline dispatching rules, print summary table, export CSV,
    and generate the Gantt chart.
    """
    env = GreenFJSPEnv(
        benchmark_path=benchmark_path,
        config_path=config_path,
        weights=(0.4, 0.4, 0.2),
        seed=42,
    )

    # 1. Rule 1: SPT + Fast Speed
    res_spt = execute_deterministic_rule(env, rule_spt_fast, seed=42)
    res_spt["rule_name"] = "Rule 1 (SPT + Fast Speed)"

    # 2. Rule 2: EET + Eco Speed
    res_eet = execute_deterministic_rule(env, rule_eet_eco, seed=42)
    res_eet["rule_name"] = "Rule 2 (EET + Eco Speed)"

    # 3. Rule 3: EDD + Balanced Speed
    res_edd = execute_deterministic_rule(env, rule_edd_balanced, seed=42)
    res_edd["rule_name"] = "Rule 3 (EDD + Balanced Speed)"

    # 4. Rule 4: Random Valid Policy (10 seeds)
    res_rand = execute_random_baseline(env, num_seeds=10)
    res_rand["rule_name"] = "Rule 4 (Random Valid Policy - 10 seeds)"

    all_results = [res_spt, res_eet, res_edd, res_rand]

    # Save to CSV
    csv_path = Path(csv_output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Rule Name",
            "Makespan (min)",
            "Total Energy (kWh)",
            "E_proc (kWh)",
            "E_idle_stdby (kWh)",
            "E_on_off (kWh)",
            "TOU Energy Cost ($)",
            "Total Weighted Tardiness (min)",
            "PM Replacements",
            "Tool Breakdowns",
        ])
        for r in all_results:
            writer.writerow([
                r["rule_name"],
                f"{r['cmax']:.2f}",
                f"{r['e_total']:.3f}",
                f"{r['e_proc']:.3f}",
                f"{r['e_idle_stdby']:.3f}",
                f"{r['e_on_off']:.3f}",
                f"{r['cost']:.3f}",
                f"{r['twt']:.2f}",
                f"{r['pms']:.1f}" if isinstance(r['pms'], float) else r['pms'],
                f"{r['bds']:.1f}" if isinstance(r['bds'], float) else r['bds'],
            ])

    # Plot Gantt Chart for Rule 3 (EDD + Balanced)
    plot_gantt_and_power_profile(
        scheduled_events=res_edd["scheduled_events"],
        machines=env.machines,
        cmax=res_edd["cmax"],
        p_limit_kw=env.p_limit_kw,
        tou_tiers=env.tou_pricing.tiers,
        output_path=gantt_output_path,
    )

    return all_results


def print_baseline_table(results: List[Dict[str, Any]]) -> None:
    """Print clean comparison table to Terminal."""
    print("=" * 110)
    print("Phase 1 Baseline Dispatching Rules Evaluation (Mk01_green.json - 10 Jobs, 6 Machines, 55 Ops)")
    print("=" * 110)
    print(
        f"{'Rule Name':<34} | "
        f"{'Cmax (min)':<10} | "
        f"{'E_tot (kWh)':<11} | "
        f"{'E_proc':<7} | "
        f"{'E_sb':<6} | "
        f"{'E_on':<6} | "
        f"{'Cost ($)':<9} | "
        f"{'TWT (min)':<9} | "
        f"{'PM / BD':<8}"
    )
    print("-" * 110)

    for r in results:
        pm_bd_str = (
            f"{r['pms']:.0f}/{r['bds']:.0f}"
            if isinstance(r['pms'], (int, np.integer))
            else f"{r['pms']:.1f}/{r['bds']:.1f}"
        )
        print(
            f"{r['rule_name']:<34} | "
            f"{r['cmax']:<10.2f} | "
            f"{r['e_total']:<11.3f} | "
            f"{r['e_proc']:<7.2f} | "
            f"{r['e_idle_stdby']:<6.2f} | "
            f"{r['e_on_off']:<6.2f} | "
            f"{r['cost']:<9.3f} | "
            f"{r['twt']:<9.2f} | "
            f"{pm_bd_str:<8}"
        )
    print("=" * 110)


if __name__ == "__main__":
    results = run_all_baselines()
    print_baseline_table(results)
    print(f"\n[OK] Results saved to CSV: data/benchmarks/phase1_baseline_results.csv")
    print(f"[OK] Gantt & Power chart generated: tests/phase1_sample_gantt.png")

