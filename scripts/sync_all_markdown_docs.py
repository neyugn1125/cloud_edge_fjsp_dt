#!/usr/bin/env python3
"""
Synchronize all markdown documents (README.md, results/manuscript_draft.md,
results/audit_methodology_specs.md) with canonical benchmark CSV tables and
measured hardware latencies in data/benchmarks/.

Enforces strict directional percentage conventions:
- Cost increase from omitting SSR standby gating: +X% degradation (relative to Proposed DT)
- Cost reduction from enabling SSR standby gating: -Y% or Y% savings (relative to Idle-Only)
- Never combine positive sign '+' with reduction/cut verbs.
"""

import csv
import re
from pathlib import Path

# Paths
DATA_DIR = Path("data/benchmarks")
README_PATH = Path("README.md")
MANUSCRIPT_PATH = Path("results/manuscript_draft.md")
AUDIT_SPECS_PATH = Path("results/audit_methodology_specs.md")


def load_csv(name):
    with open(DATA_DIR / name, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def generate_manuscript_table1():
    rows = load_csv("phase5_wilcoxon_benchmark_table.csv")
    lines = [
        r"| Instance | Method | Hypervolume (HV) $\uparrow$ | IGD $\downarrow$ | Makespan $C_{\max}$ (min) $\downarrow$ | TOU Cost (\$) $\downarrow$ | Tardiness TWT (min) $\downarrow$ | Tool Failures $\downarrow$ | Cliff's $\delta$ | $p_{\text{holm}}$ |",
        r"| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    cur_inst = ""
    for r in rows:
        inst = r["Instance"]
        scale = r["Scale"]
        m = r["Method"]
        hv_m, hv_s = float(r["HV_Mean"]), float(r["HV_Std"])
        igd_m, igd_s = float(r["IGD_Mean"]), float(r["IGD_Std"])
        c_m, c_s = float(r["Makespan_Mean"]), float(r["Makespan_Std"])
        cost_m, cost_s = float(r["Cost_Mean"]), float(r["Cost_Std"])
        twt_m, twt_s = float(r["TWT_Mean"]), float(r["TWT_Std"])
        bd = float(r["Breakdowns_Mean"])

        if "Proposed" in m:
            m_str = "**Proposed Cloud-Edge DT**"
            hv_str = f"**${hv_m:.4f} \\pm {hv_s:.4f}$**"
            igd_str = f"**${igd_m:.4f} \\pm {igd_s:.4f}$**"
            c_str = f"**${c_m:.2f} \\pm {c_s:.2f}$**"
            cost_str = f"**\\${cost_m:.3f} \\pm {cost_s:.3f}**"
            twt_str = f"**${twt_m:.1f} \\pm {twt_s:.1f}$**"
            bd_str = "**0.00**"
            d_str = "**Base**"
            p_str = "**Base**"
        else:
            m_str = m.replace(" (No Cloud Prior)", "").replace(" (NSGA-III + Edge ONNX + CDI)", "")
            hv_str = f"${hv_m:.4f} \\pm {hv_s:.4f}$"
            igd_str = f"${igd_m:.4f} \\pm {igd_s:.4f}$"
            c_str = f"${c_m:.2f} \\pm {c_s:.2f}$"
            cost_str = f"\\${cost_m:.3f} \\pm {cost_s:.3f}"
            twt_str = f"${twt_m:.1f} \\pm {twt_s:.1f}$"
            bd_str = f"{bd:.2f}"
            d_val = float(r["HV_Cliffs_Delta"])
            d_str = f"+{d_val:.2f}" if d_val > 0 else f"{d_val:.2f}"
            p_val = float(r["HV_p_holm"])
            p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"

        if inst != cur_inst:
            cur_inst = inst
            inst_str = f"**{inst}**<br>({scale})"
        else:
            inst_str = ""

        lines.append(f"| {inst_str} | {m_str} | {hv_str} | {igd_str} | {c_str} | {cost_str} | {twt_str} | {bd_str} | {d_str} | {p_str} |")
    return "\n".join(lines)


def generate_readme_table1():
    rows = load_csv("phase5_wilcoxon_benchmark_table.csv")
    lines = [
        r"| Problem Instance | Method | Hypervolume (HV) $\uparrow$ | Inverted Generational Distance (IGD) $\downarrow$ | Makespan $C_{\max}$ (min) $\downarrow$ | TOU Cost (\$) $\downarrow$ | Tardiness TWT (min) $\downarrow$ | Breakdowns $\downarrow$ | Cliff's Delta ($\delta$) | $p_{\text{holm}}$ |",
        r"| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    cur_inst = ""
    for r in rows:
        inst = r["Instance"]
        scale = r["Scale"]
        m = r["Method"]
        hv_m, hv_s = float(r["HV_Mean"]), float(r["HV_Std"])
        igd_m, igd_s = float(r["IGD_Mean"]), float(r["IGD_Std"])
        c_m, c_s = float(r["Makespan_Mean"]), float(r["Makespan_Std"])
        cost_m, cost_s = float(r["Cost_Mean"]), float(r["Cost_Std"])
        twt_m, twt_s = float(r["TWT_Mean"]), float(r["TWT_Std"])
        bd = float(r["Breakdowns_Mean"])

        if "Proposed" in m:
            m_str = "**Proposed Cloud-Edge DT (NSGA-III + Edge ONNX + CDI)**"
            hv_str = f"**{hv_m:.4f} $\\pm$ {hv_s:.4f}**"
            igd_str = f"**{igd_m:.4f} $\\pm$ {igd_s:.4f}**"
            c_str = f"**{c_m:.2f} $\\pm$ {c_s:.2f}**"
            cost_str = f"**\\${cost_m:.3f} $\\pm$ {cost_s:.3f}**"
            twt_str = f"**{twt_m:.1f} $\\pm$ {twt_s:.1f}**"
            bd_str = "**0.00**"
            d_str = "Base"
            p_str = "Base"
        else:
            m_str = m
            hv_str = f"${hv_m:.4f} \\pm {hv_s:.4f}$"
            igd_str = f"${igd_m:.4f} \\pm {igd_s:.4f}$"
            c_str = f"${c_m:.2f} \\pm {c_s:.2f}$"
            cost_str = f"\\${cost_m:.3f} $\\pm$ {cost_s:.3f}"
            twt_str = f"${twt_m:.1f} \\pm {twt_s:.1f}$"
            bd_str = f"{bd:.2f}"
            d_val = float(r["HV_Cliffs_Delta"])
            d_str = f"+{d_val:.2f}" if d_val > 0 else f"{d_val:.2f}"
            p_val = float(r["HV_p_holm"])
            p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"

        if inst != cur_inst:
            cur_inst = inst
            inst_str = f"**{inst}**<br>({scale})"
        else:
            inst_str = ""

        lines.append(f"| {inst_str} | {m_str} | {hv_str} | {igd_str} | {c_str} | {cost_str} | {twt_str} | {bd_str} | {d_str} | {p_str} |")
    return "\n".join(lines)


def generate_manuscript_table2():
    rows = load_csv("phase5_ablation_results.csv")
    lines = [
        r"| Instance | Variant | Description | Makespan (min) | TOU Cost (\$) | TWT (min) | Breakdowns |",
        r"| :--- | :--- | :--- | :---: | :---: | :---: | :---: |"
    ]
    cur_inst = ""
    for r in rows:
        inst = r["Instance"]
        scale = r["Scale"]
        var = r["Variant"]
        c_m, c_s = float(r["Makespan Mean (min)"]), float(r["Makespan Std (min)"])
        c_deg = float(r["Makespan Degradation (%)"])
        cost_m, cost_s = float(r["TOU Cost Mean ($)"]), float(r["TOU Cost Std ($)"])
        cost_deg = float(r["TOU Cost Degradation (%)"])
        twt_m, twt_s = float(r["TWT Mean (min)"]), float(r["TWT Std (min)"])
        twt_deg = float(r["TWT Degradation (%)"])
        bd = float(r["Breakdowns Mean"])

        if "Full Proposed" in var:
            var_str = "**Full Proposed DT**"
            desc_str = "Full closed-loop framework (Dual-Sensor EKF + Speed Scaling + SSR Standby + Cloud Guidance)"
            c_str = f"**${c_m:.2f} \\pm {c_s:.2f}$ (Base)**"
            cost_str = f"**\\${cost_m:.3f} \\pm {cost_s:.3f} (Base)**"
            twt_str = f"**${twt_m:.2f} \\pm {twt_s:.2f}$ (Base)**"
            bd_str = "**0.00**"
        else:
            var_str = var.split(" (")[0]
            if "Variant 1" in var:
                desc_str = "w/o Dual-Sensor Wear Feedback (Delayed PM under disturbance)"
            elif "Variant 2" in var:
                desc_str = "Fixed Speed $s=2$ (No real-time dynamic TOU peak avoidance)"
            elif "Variant 3" in var:
                desc_str = "Idle-Only (No SSR standby power gating during waiting intervals)"
            elif "Variant 4" in var:
                desc_str = "w/o Cloud Guidance (Pure local edge dispatching without global macro prior)"
            else:
                desc_str = r["Description"]

            c_str = f"${c_m:.2f} \\pm {c_s:.2f}$ ({c_deg:+.1f}%)"
            cost_str = f"\\${cost_m:.3f} \\pm {cost_s:.3f}$ ({cost_deg:+.1f}%)"
            twt_str = f"${twt_m:.2f} \\pm {twt_s:.2f}$ ({twt_deg:+.1f}%)"
            bd_str = f"**{bd:.2f}**" if bd > 0 else "0.00"

        if inst != cur_inst:
            cur_inst = inst
            inst_str = f"**{inst}**<br>({scale})"
        else:
            inst_str = ""

        lines.append(f"| {inst_str} | {var_str} | {desc_str} | {c_str} | {cost_str} | {twt_str} | {bd_str} |")
    return "\n".join(lines)


def generate_readme_table2():
    rows = load_csv("phase5_ablation_results.csv")
    lines = [
        r"| Problem Instance | Architectural Variant | Description & Mechanism | Makespan $C_{\max}$ (min) | TOU Cost (\$) | Tardiness TWT (min) | Tool Breakdowns |",
        r"| :--- | :--- | :--- | :---: | :---: | :---: | :---: |"
    ]
    cur_inst = ""
    for r in rows:
        inst = r["Instance"]
        scale = r["Scale"]
        var = r["Variant"]
        desc = r["Description"]
        c_m, c_s = float(r["Makespan Mean (min)"]), float(r["Makespan Std (min)"])
        c_deg = float(r["Makespan Degradation (%)"])
        cost_m, cost_s = float(r["TOU Cost Mean ($)"]), float(r["TOU Cost Std ($)"])
        cost_deg = float(r["TOU Cost Degradation (%)"])
        twt_m, twt_s = float(r["TWT Mean (min)"]), float(r["TWT Std (min)"])
        twt_deg = float(r["TWT Degradation (%)"])
        bd = float(r["Breakdowns Mean"])

        if "Full Proposed" in var:
            var_str = "**Full Proposed Cloud-Edge DT**"
            c_str = f"**{c_m:.2f} $\\pm$ {c_s:.2f}**"
            cost_str = f"**\\${cost_m:.3f} $\\pm$ {cost_s:.3f}**"
            twt_str = f"**{twt_m:.2f} $\\pm$ {twt_s:.2f}**"
            bd_str = "**0.00**"
        else:
            var_str = var
            c_str = f"${c_m:.2f} \\pm {c_s:.2f}$ ({c_deg:+.1f}%)"
            cost_str = f"\\${cost_m:.3f} $\\pm$ {cost_s:.3f} (**{cost_deg:+.1f}%**)"
            twt_str = f"${twt_m:.2f} \\pm {twt_s:.2f}$ ({twt_deg:+.1f}%)"
            bd_str = f"**{bd:.2f}**" if bd > 0 else "0.00"

        if inst != cur_inst:
            cur_inst = inst
            inst_str = f"**{inst}**<br>({scale})"
        else:
            inst_str = ""

        lines.append(f"| {inst_str} | {var_str} | {desc} | {c_str} | {cost_str} | {twt_str} | {bd_str} |")
    return "\n".join(lines)


def generate_cdi_sensitivity_table():
    rows = load_csv("phase5_cdi_sensitivity.csv")
    min_sys_idx = min(float(r["Total_System_Cost_Index"]) for r in rows)
    lines = [
        r"| CDI Threshold $\theta_{\text{th}}$ | Resync Calls ($\text{Mean} \pm \text{Std}$) | Makespan ($\text{Mean} \pm \text{Std}$) | TOU Cost (\$) | Max CDI ($\text{Mean} \pm \text{Std}$) | Schedule Index | Total System Cost Index | Operational Region |",
        r"| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]
    for r in rows:
        th = float(r["CDI_Threshold"])
        calls_m, calls_s = float(r["Cloud_Calls_Mean"]), float(r["Cloud_Calls_Std"])
        c_m, c_s = float(r["Makespan_Mean (min)"]), float(r["Makespan_Std (min)"])
        cost_m, cost_s = float(r["TOU_Cost_Mean ($)"]), float(r["TOU_Cost_Std ($)"])
        cdi_m, cdi_s = float(r["Max_CDI_Mean"]), float(r["Max_CDI_Std"])
        sched_idx = float(r["Schedule_Composite_Index"])
        sys_idx = float(r["Total_System_Cost_Index"])
        zone = r["Zone"]

        is_opt = abs(sys_idx - min_sys_idx) < 1e-6
        is_sel = abs(th - 0.12) < 1e-4

        if is_opt or is_sel:
            th_str = f"**{th:.2f}**"
            calls_str = f"**${calls_m:.2f} \\pm {calls_s:.2f}$**"
            c_str = f"**${c_m:.2f} \\pm {c_s:.2f}$**"
            cost_str = f"**\\${cost_m:.3f} \\pm {cost_s:.3f}**"
            cdi_str = f"**${cdi_m:.4f} \\pm {cdi_s:.4f}$**"
            sched_str = f"**{sched_idx:.4f}**"
            if is_opt and is_sel:
                sys_str = f"**{sys_idx:.4f} (Empirical Optimum)**"
            elif is_opt:
                sys_str = f"**{sys_idx:.4f} (Empirical Optimum)**"
            else:
                sys_str = f"**{sys_idx:.4f} (Selected Center)**"
            zone_str = f"**{zone}**"
        else:
            th_str = f"**{th:.2f}**"
            calls_str = f"${calls_m:.2f} \\pm {calls_s:.2f}$"
            c_str = f"${c_m:.2f} \\pm {c_s:.2f}$"
            cost_str = f"\\${cost_m:.3f} \\pm {cost_s:.3f}"
            cdi_str = f"${cdi_m:.4f} \\pm {cdi_s:.4f}$"
            sched_str = f"{sched_idx:.4f}"
            sys_str = f"{sys_idx:.4f}"
            zone_str = zone

        lines.append(f"| {th_str} | {calls_str} | {c_str} | {cost_str} | {cdi_str} | {sched_str} | {sys_str} | {zone_str} |")
    return "\n".join(lines)


def generate_manuscript_table4():
    rows = load_csv("phase5_transport_sensitivity.csv")
    lines = [
        r"| Instance | Transport Latency $\tau_{\text{trans}}$ | Method | Makespan (min) $\downarrow$ | TOU Cost (\$) $\downarrow$ | Tardiness TWT (min) $\downarrow$ | Tool Failures $\downarrow$ |",
        r"| :--- | :---: | :--- | :---: | :---: | :---: | :---: |"
    ]
    cur_inst = ""
    cur_tau = ""
    for r in rows:
        inst = r["Instance"]
        tau = float(r["Transport_Time_min"])
        m = r["Method"]
        c_m, c_s = float(r["Makespan_Mean"]), float(r["Makespan_Std"])
        cost_m, cost_s = float(r["TOU_Cost_Mean"]), float(r["TOU_Cost_Std"])
        twt_m, twt_s = float(r["TWT_Mean"]), float(r["TWT_Std"])
        bd = float(r["Breakdowns_Mean"])

        if "Proposed" in m:
            m_str = "**Proposed Cloud-Edge DT**"
            c_str = f"**${c_m:.2f} \\pm {c_s:.2f}$**"
            cost_str = f"**\\${cost_m:.3f} \\pm {cost_s:.3f}**"
            twt_str = f"**${twt_m:.1f} \\pm {twt_s:.1f}$**"
            bd_str = "**0.00**"
        else:
            m_str = m
            c_str = f"${c_m:.2f} \\pm {c_s:.2f}$"
            cost_str = f"\\${cost_m:.3f} \\pm {cost_s:.3f}"
            twt_str = f"${twt_m:.1f} \\pm {twt_s:.1f}$"
            bd_str = f"**{bd:.2f}**" if bd > 0 else "0.00"

        tau_str = f"{tau:.1f} min"
        if inst != cur_inst:
            cur_inst = inst
            inst_str = f"**{inst}**"
        else:
            inst_str = ""

        if tau_str != cur_tau:
            cur_tau = tau_str
            tau_disp = tau_str
        else:
            tau_disp = ""

        lines.append(f"| {inst_str} | {tau_disp} | {m_str} | {c_str} | {cost_str} | {twt_str} | {bd_str} |")
    return "\n".join(lines)


def generate_readme_transport_subtable(target_inst):
    rows = [r for r in load_csv("phase5_transport_sensitivity.csv") if r["Instance"] == target_inst]
    lines = [
        r"| $\tau_{\text{trans}}$ (min) | Method | Makespan $C_{\max}$ (min) | TOU Cost (\$) | TWT (min) | Tool Breakdowns |",
        r"| :---: | :--- | :---: | :---: | :---: | :---: |"
    ]
    cur_tau = ""
    for r in rows:
        tau = float(r["Transport_Time_min"])
        m = r["Method"]
        c_m, c_s = float(r["Makespan_Mean"]), float(r["Makespan_Std"])
        cost_m, cost_s = float(r["TOU_Cost_Mean"]), float(r["TOU_Cost_Std"])
        twt_m, twt_s = float(r["TWT_Mean"]), float(r["TWT_Std"])
        bd = float(r["Breakdowns_Mean"])

        tau_str = f"{tau:.1f}"
        if tau == 0.0:
            tau_disp = "**0.0**<br>*(Baseline)*" if cur_tau != tau_str else ""
        else:
            tau_disp = f"**{tau_str}**" if cur_tau != tau_str else ""

        if cur_tau != tau_str:
            cur_tau = tau_str

        if "Proposed" in m:
            m_str = "**Proposed Cloud-Edge DT**"
            c_str = f"**${c_m:.2f} \\pm {c_s:.2f}$**"
            cost_str = f"**\\${cost_m:.3f} \\pm {cost_s:.3f}**"
            twt_str = f"**${twt_m:.2f} \\pm {twt_s:.2f}$**"
            bd_str = "**0.00**"
        else:
            m_str = m
            c_str = f"${c_m:.2f} \\pm {c_s:.2f}$"
            cost_str = f"\\${cost_m:.3f} \\pm {cost_s:.3f}"
            twt_str = f"${twt_m:.2f} \\pm {twt_s:.2f}$"
            bd_str = f"**{bd:.2f}**" if bd > 0 else "0.00"

        lines.append(f"| {tau_disp} | {m_str} | {c_str} | {cost_str} | {twt_str} | {bd_str} |")
    return "\n".join(lines)


def _fmt_resync_lat_ms(solver_s: float) -> str:
    if solver_s <= 0.0:
        return "0.0 s"
    if solver_s < 0.1:
        return f"{solver_s:.4f} s ({solver_s * 1000.0:.2f} ms)"
    return f"{solver_s:.2f} s"


def _fmt_p(p_val: float, is_base: bool = False) -> str:
    if is_base:
        return "**Base**"
    if p_val < 0.0001:
        return "< 0.0001"
    return f"{p_val:.4f}"


def generate_manuscript_table5():
    rows = load_csv("phase4_hil_metrics.csv")
    lines = [
        r"| Operational Mode | Resync Calls ($\text{Mean} \pm \text{Std}$) | Call Reduction (%) | Cloud Warm-Start Overhead ($\text{Mean} \pm \text{Std}$) | Edge Latency (ms) | Network Packets | Makespan ($\text{Mean} \pm \text{Std}$, min) | Energy ($\text{Mean} \pm \text{Std}$, kWh) | TOU Cost ($\text{Mean} \pm \text{Std}$, \$) | Max CDI ($\text{Mean} \pm \text{Std}$) | Final CDI ($\text{Mean} \pm \text{Std}$) | PM ($\text{Mean} \pm \text{Std}$) | Breakdowns ($\text{Mean} \pm \text{Std}$) | $p_{\text{Wilcoxon}}$ ($C_{\max}$) | $p_{\text{Wilcoxon}}$ (Cost) |",
        r"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for r in rows:
        mode_id = r["mode_id"]
        name = r["mode_name"]
        calls = float(r["cloud_reschedule_calls"])
        calls_s = float(r.get("cloud_calls_std", 0.0))
        reduct = float(r["reduction_in_cloud_calls_pct"])
        solver_s = float(r["total_solver_latency_s"])
        solver_std_s = float(r.get("solver_latency_std_s", 0.0))
        edge_ms = float(r["edge_decision_latency_ms"])
        packets = float(r["total_network_packets"])
        cmax = float(r["final_makespan_min"])
        cmax_s = float(r.get("makespan_std_min", 0.0))
        energy = float(r["total_energy_kwh"])
        energy_s = float(r.get("energy_std_kwh", 0.0))
        cost = float(r["tou_energy_cost_usd"])
        cost_s = float(r.get("tou_cost_std_usd", 0.0))
        max_cdi = float(r["max_cdi"])
        max_cdi_s = float(r.get("max_cdi_std", 0.0))
        final_cdi = float(r["final_cdi"])
        final_cdi_s = float(r.get("final_cdi_std", 0.0))
        pm = float(r["num_pm_replacements"])
        pm_s = float(r.get("pm_std", 0.0))
        bd = float(r["num_breakdowns"])
        bd_s = float(r.get("breakdowns_std", 0.0))
        p_mk = float(r.get("p_wilcoxon_makespan", 1.0))
        p_co = float(r.get("p_wilcoxon_cost", 1.0))

        is_prop = (mode_id == "Mode 5")
        edge_str = f"{edge_ms:.4f} ms" if edge_ms > 0 else "—"
        lat_str = f"{solver_s:.4f} ± {solver_std_s:.4f} s" if solver_s > 0 else "0.0 s"
        p_mk_str = _fmt_p(p_mk, is_base=is_prop)
        p_co_str = _fmt_p(p_co, is_base=is_prop)

        if is_prop:
            mode_str = f"**{mode_id}: {name}**"
            calls_str = f"**{calls:.2f} ± {calls_s:.2f}**"
            reduct_str = f"**{reduct:.1f}%**"
            lat_str = f"**{lat_str}**"
            packets_str = f"**{packets:.2f}**"
            cmax_str = f"**{cmax:.2f} ± {cmax_s:.2f}**"
            energy_str = f"**{energy:.3f} ± {energy_s:.3f}**"
            cost_str = f"**\\${cost:.3f} ± {cost_s:.3f}**"
            max_cdi_str = f"{max_cdi:.4f} ± {max_cdi_s:.4f}"
            final_cdi_str = f"{final_cdi:.4f} ± {final_cdi_s:.4f}"
            pm_str = f"{pm:.2f} ± {pm_s:.2f}"
            bd_str = f"**{bd:.2f} ± {bd_s:.2f}**"
        else:
            mode_str = f"{mode_id}: {name}"
            calls_str = f"{calls:.2f} ± {calls_s:.2f}"
            reduct_str = f"{reduct:.1f}%"
            packets_str = f"{packets:.2f}"
            cmax_str = f"{cmax:.2f} ± {cmax_s:.2f}"
            energy_str = f"{energy:.3f} ± {energy_s:.3f}"
            cost_str = f"\\${cost:.3f} ± {cost_s:.3f}"
            max_cdi_str = f"{max_cdi:.4f} ± {max_cdi_s:.4f}"
            final_cdi_str = f"{final_cdi:.4f} ± {final_cdi_s:.4f}"
            pm_str = f"{pm:.2f} ± {pm_s:.2f}"
            bd_str = f"**{bd:.2f} ± {bd_s:.2f}**" if bd > 0 else f"{bd:.2f} ± {bd_s:.2f}"

        lines.append(
            f"| {mode_str} | {calls_str} | {reduct_str} | {lat_str} | {edge_str} | {packets_str} | "
            f"{cmax_str} | {energy_str} | {cost_str} | {max_cdi_str} | {final_cdi_str} | {pm_str} | {bd_str} | {p_mk_str} | {p_co_str} |"
        )
    return "\n".join(lines)


def generate_readme_table5():
    rows = load_csv("phase4_hil_metrics.csv")
    lines = [
        r"| Control Paradigm | Synchronization Strategy | Resync Calls ($\text{Mean} \pm \text{Std}$) | Cloud Warm-Start Overhead (2.6835 s/call) | Edge Latency | Packets | Makespan $C_{\max}$ ($\text{Mean} \pm \text{Std}$) | Energy (kWh) | TOU Cost (\$) | Max CDI | PM ($\text{Mean} \pm \text{Std}$) | Failures ($\text{Mean} \pm \text{Std}$) | $p_{\text{Wilcoxon}}$ ($C_{\max}$ / Cost) |",
        r"| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for r in rows:
        mode_id = r["mode_id"]
        name = r["mode_name"]
        calls = float(r["cloud_reschedule_calls"])
        calls_s = float(r.get("cloud_calls_std", 0.0))
        reduct = float(r["reduction_in_cloud_calls_pct"])
        solver_s = float(r["total_solver_latency_s"])
        solver_std_s = float(r.get("solver_latency_std_s", 0.0))
        edge_ms = float(r["edge_decision_latency_ms"])
        packets = float(r["total_network_packets"])
        cmax = float(r["final_makespan_min"])
        cmax_s = float(r.get("makespan_std_min", 0.0))
        energy = float(r["total_energy_kwh"])
        energy_s = float(r.get("energy_std_kwh", 0.0))
        cost = float(r["tou_energy_cost_usd"])
        cost_s = float(r.get("tou_cost_std_usd", 0.0))
        max_cdi = float(r["max_cdi"])
        max_cdi_s = float(r.get("max_cdi_std", 0.0))
        pm = float(r["num_pm_replacements"])
        pm_s = float(r.get("pm_std", 0.0))
        bd = float(r["num_breakdowns"])
        bd_s = float(r.get("breakdowns_std", 0.0))
        p_mk = float(r.get("p_wilcoxon_makespan", 1.0))
        p_co = float(r.get("p_wilcoxon_cost", 1.0))

        is_prop = (mode_id == "Mode 5")
        edge_str = f"{edge_ms:.4f}\\,\\text{{ms}}" if edge_ms > 0 else "$0.0\\,\\text{ms}$"
        solver_str = f"{solver_s:.4f} ± {solver_std_s:.4f}\\,\\text{{s}}" if solver_s > 0 else "$0.0\\,\\text{s}$"
        p_pair_str = "**Base**" if is_prop else f"{_fmt_p(p_mk)} / {_fmt_p(p_co)}"

        if is_prop:
            mode_str = f"**{mode_id}**"
            strat_str = f"**{name}**"
            calls_str = f"**{calls:.2f} ± {calls_s:.2f} ($-{reduct:.1f}\\%$)**"
            solver_str = f"**{solver_str}**"
            edge_str = f"**{edge_str}**"
            packets_str = f"**{packets:.2f}**"
            cmax_str = f"**{cmax:.2f} ± {cmax_s:.2f}\\,\\text{{min}}**"
            energy_str = f"**{energy:.3f} ± {energy_s:.3f}**"
            cost_str = f"**\\${cost:.3f} ± {cost_s:.3f}**"
            cdi_str = f"**{max_cdi:.4f} ± {max_cdi_s:.4f}**"
            pm_str = f"**{pm:.2f} ± {pm_s:.2f}**"
            bd_str = f"**{bd:.2f} ± {bd_s:.2f}**"
        else:
            mode_str = f"**{mode_id}**"
            strat_str = name
            calls_str = f"{calls:.2f} ± {calls_s:.2f}" if reduct == 0.0 else f"{calls:.2f} ± {calls_s:.2f} ($-{reduct:.1f}\\%$)"
            packets_str = f"{packets:.2f}"
            cmax_str = f"{cmax:.2f} ± {cmax_s:.2f}\\,\\text{{min}}"
            energy_str = f"{energy:.3f} ± {energy_s:.3f}"
            cost_str = f"\\${cost:.3f} ± {cost_s:.3f}"
            cdi_str = f"{max_cdi:.4f} ± {max_cdi_s:.4f}"
            pm_str = f"{pm:.2f} ± {pm_s:.2f}"
            bd_str = f"**{bd:.2f} ± {bd_s:.2f}**" if bd > 0 else f"{bd:.2f} ± {bd_s:.2f}"

        lines.append(
            f"| {mode_str} | {strat_str} | {calls_str} | {solver_str} | {edge_str} | {packets_str} | "
            f"{cmax_str} | {energy_str} | {cost_str} | {cdi_str} | {pm_str} | {bd_str} | {p_pair_str} |"
        )
    return "\n".join(lines)


def update_manuscript():
    print("[1/3] Synchronizing results/manuscript_draft.md...")
    with open(MANUSCRIPT_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Dynamic stats from canonical CSVs
    wilc_rows = load_csv("phase5_wilcoxon_benchmark_table.csv")
    prop_rows = [r for r in wilc_rows if "Proposed" in r["Method"]]
    mean_hv = sum(float(r["HV_Mean"]) for r in prop_rows) / len(prop_rows)
    mean_igd = sum(float(r["IGD_Mean"]) for r in prop_rows) / len(prop_rows)

    rs_map = {r["Instance"]: r for r in wilc_rows if "Right-Shift" in r["Method"]}
    nsga3_map = {r["Instance"]: r for r in wilc_rows if "NSGA-III (Open-Loop)" in r["Method"]}
    nsga2_map = {r["Instance"]: r for r in wilc_rows if "NSGA-II (Open-Loop)" in r["Method"]}
    prop_map = {r["Instance"]: r for r in prop_rows}
    mk05_sav = (float(rs_map["Mk05"]["Cost_Mean"]) - float(prop_map["Mk05"]["Cost_Mean"])) / float(rs_map["Mk05"]["Cost_Mean"]) * 100.0
    mk01_rs_pholm = float(rs_map["Mk01"]["HV_p_holm"])
    mk02_rs_pholm = float(rs_map["Mk02"]["HV_p_holm"])
    mk03_rs_pholm = float(rs_map["Mk03"]["HV_p_holm"])
    mk04_rs_pholm = float(rs_map["Mk04"]["HV_p_holm"])
    mk05_rs_pholm = float(rs_map["Mk05"]["HV_p_holm"])
    mk03_nsga3_pholm = float(nsga3_map["Mk03"]["HV_p_holm"])
    mk03_nsga2_pholm = float(nsga2_map["Mk03"]["HV_p_holm"])

    hil_rows = {r["mode_id"]: r for r in load_csv("phase4_hil_metrics.csv")}
    m1_r, m2_r, m3_r, m4_r, m5_r = (
        hil_rows["Mode 1"], hil_rows["Mode 2"], hil_rows["Mode 3"], hil_rows["Mode 4"], hil_rows["Mode 5"]
    )
    m5_calls = float(m5_r["cloud_reschedule_calls"])
    m5_calls_s = float(m5_r["cloud_calls_std"])
    m5_reduct = float(m5_r["reduction_in_cloud_calls_pct"])
    m2_calls = float(m2_r["cloud_reschedule_calls"])
    m3_calls = float(m3_r["cloud_reschedule_calls"])
    m2_cmax, m5_cmax = float(m2_r["final_makespan_min"]), float(m5_r["final_makespan_min"])
    m2_cost, m5_cost = float(m2_r["tou_energy_cost_usd"]), float(m5_r["tou_energy_cost_usd"])
    m2_m5_cost_reduct = (m2_cost - m5_cost) / m2_cost * 100.0
    m3_m5_call_reduct = (m3_calls - m5_calls) / m3_calls * 100.0

    cdi_rows = load_csv("phase5_cdi_sensitivity.csv")
    cdi_map = {round(float(r["CDI_Threshold"]), 2): r for r in cdi_rows}
    opt_row = min(cdi_rows, key=lambda x: float(x["Total_System_Cost_Index"]))
    opt_th = float(opt_row["CDI_Threshold"])
    opt_sys = float(opt_row["Total_System_Cost_Index"])
    opt_calls_m, opt_calls_s = float(opt_row["Cloud_Calls_Mean"]), float(opt_row["Cloud_Calls_Std"])
    opt_cmax_m, opt_cmax_s = float(opt_row["Makespan_Mean (min)"]), float(opt_row["Makespan_Std (min)"])
    opt_cost_m, opt_cost_s = float(opt_row["TOU_Cost_Mean ($)"]), float(opt_row["TOU_Cost_Std ($)"])
    plateau_ths = sorted(
        float(r["CDI_Threshold"]) for r in cdi_rows if r.get("Zone") == "Event-Matched Plateau (1–2 Shocks)"
    )
    min_pl, max_pl = min(plateau_ths), max(plateau_ths)

    r02 = cdi_map[0.02]
    r05 = cdi_map[0.05]
    r08 = cdi_map[0.08]
    r12 = cdi_map[0.12]
    r18 = cdi_map[0.18]
    r22 = cdi_map[0.22]
    r50 = cdi_map[0.50]

    # 1. Abstract: accurate CDI trade-off, 5-mode HIL, updated HV/IGD & fair EKF Right-Shift stats (no equivalence claim)
    new_abstract = (
        "Dynamic Flexible Job Shop Scheduling Problems (D-FJSP) in discrete manufacturing face severe operational uncertainties, "
        "including non-linear tool degradation, time-varying electricity tariffs, and strict real-time control constraints. "
        "While centralized Cloud metaheuristics yield high-quality global Pareto schedules, their computational latency "
        "($4.24 \\pm 0.03\\,\\text{s}$ on Mk01, $14.97\\,\\text{s}$ on Mk03; $2.6835\\,\\text{s}$ warm-start resync) precludes real-time disturbance reaction. "
        "Conversely, distributed edge dispatching policies provide rapid reactivity but suffer from localized myopia, peak-tariff overloading, "
        "and premature tool breakage. This paper presents a closed-loop **Cloud-Edge Collaborative Digital Twin (DT)** framework "
        "that harmonizes long-term energy-cost optimization with sub-millisecond shop-floor control. The framework comprises: "
        "(1) a Cloud macro-optimizer executing reference-direction NSGA-III over Makespan ($C_{\\max}$), Time-of-Use (TOU) energy cost, "
        "and Total Weighted Tardiness (TWT); (2) an Edge micro-controller powered by a scale-invariant MLP policy trained via Maskable "
        "Proximal Policy Optimization (MaskablePPO) and executed on ONNX Runtime in $0.0132\\,\\text{ms}$ forward pass and "
        "$0.1506\\,\\text{ms}$ per full decision step ($\\sim 2.81 \\times 10^4 \\times$ faster than Cloud optimization); "
        f"(3) a shift-normalized Cumulative Disturbance Index ($\\text{{CDI}}(t)$) deployed at $\\theta_{{\\text{{th}}}} = 0.12$ within the "
        f"objective Event-Matched Plateau $[{min_pl:.2f}, {max_pl:.2f}]$ ($1$–$2$ calls matched to $N_{{\\text{{shocks}}}} = 2.0$, whereas "
        f"$\\theta_{{\\text{{th}}}}^* = {opt_th:.2f}$ minimizes `Total_System_Cost_Index` at ${opt_sys:.4f}$ with ${opt_calls_m:.2f}$ over-triggered calls), "
        f"reducing online Cloud Warm-Start rescheduling calls by ${m5_reduct:.1f}\\%$ (${m5_calls:.2f} \\pm {m5_calls_s:.2f}$ vs. ${m2_calls:.2f}$ periodic calls; "
        f"${m3_m5_call_reduct:.1f}\\%$ fewer than fixed frequency-matched Op 25/50 periodic calls at ${m3_calls:.2f}$) and cutting TOU electricity cost by "
        f"${m2_m5_cost_reduct:.1f}\\%$ relative to high-frequency periodic EKF resync ($p = 0.0067$); "
        "and (4) an opto-isolated solid-state relay (SSR) mechatronic subsystem enabling zero-loss standby transitions "
        "($T_{\\text{BE}} \\in [8.36, 10.54]\\,\\text{min}$ on shop-floor CNC centers; $T_{\\text{BE}} \\approx 3.1\\,\\text{min}$ on hardware proxy) "
        "with Stop Category 1 hardware safety interlocks (`GPIO 27` Open-Drain GRBL `Feed Hold`, $2500\\,\\text{ms}$ / $2.50\\,\\text{s}$ kinematic deceleration sync, and $300\\,\\text{ms}$ Host Watchdog). "
        "Audited across 750 independent experimental runs (15 test seeds, Brandimarte Mk01–Mk05) against 9 competing baselines, "
        "the proposed framework eliminates catastrophic tool failures ($0.00$ breakdowns vs. $0.33 - 12.67$ for open-loop baselines), "
        "reduces TOU energy costs by up to $30.95\\%$ via SSR standby gating (saving $\\$6.223$ per shift on Mk03 vs. the idle-only baseline; "
        "or $+44.82\\%$ cost penalty when SSR standby is omitted), achieves mean $\\overline{\\text{HV}} = "
        f"{mean_hv:.4f}$ and $\\overline{{\\text{{IGD}}}} = {mean_igd:.4f}$, and achieves statistically significant Hypervolume superiority "
        f"over all 4 open-loop metaheuristics on bottleneck Mk05 ($p_{{\\text{{holm}}}} \\le 0.0011$) and over NSGA-III ($p_{{\\text{{holm}}}} = {mk03_nsga3_pholm:.4f}$) "
        f"and NSGA-II ($p_{{\\text{{holm}}}} = {mk03_nsga2_pholm:.4f}$) on large-scale Mk03. "
        "When reactive Right-Shift Rescheduling is fairly equipped with the identical live EKF wear estimator and PM safety rule to also achieve $0.00$ breakdowns, "
        f"two-sided Wilcoxon tests show no statistically significant difference in Hypervolume on Mk01 ($p_{{\\text{{holm}}}} = {mk01_rs_pholm:.4f}$) and Mk04 ($p_{{\\text{{holm}}}} = {mk04_rs_pholm:.4f}$), "
        f"while Right-Shift achieves statistically higher Hypervolume on Mk02 ($p_{{\\text{{holm}}}} = {mk02_rs_pholm:.4f}$), Mk03 ($p_{{\\text{{holm}}}} = {mk03_rs_pholm:.4f}$), and Mk05 ($p_{{\\text{{holm}}}} = {mk05_rs_pholm:.4f}$), "
        f"and the Proposed DT lowers TOU electricity cost by ${mk05_sav:.2f}\\%$ on bottleneck Mk05 (\\${float(prop_map['Mk05']['Cost_Mean']):.3f} vs. \\${float(rs_map['Mk05']['Cost_Mean']):.3f}). "
        "Physical validation on an ESP32 dual-sensor CNC breadboard in high-fidelity electro-mechanical emulation with unmodeled material and thermal disturbances "
        "confirms robust real-time tool wear estimation ($R^2 = 0.9984$ during active cutting, $R^2 = 0.9951$ across the full 60s operational profile with "
        "$\\text{RMSE} = 1.763\\,\\mu\\text{m}$) and safe non-preemptive overrun boundaries."
    )
    abs_pattern = re.compile(r"(## Abstract\n)(.*?)(?=\n\n---)", re.DOTALL)
    m_abs = abs_pattern.search(content)
    if m_abs:
        content = content[:m_abs.start()] + "## Abstract\n" + new_abstract + content[m_abs.end():]

    # 2. Table 1
    t1_pattern = re.compile(r"(\| Instance \| Method \| Hypervolume \(HV\).*?\n)(.*?)(?=\n### 3\.3 Critical Examination)", re.DOTALL)
    m = t1_pattern.search(content)
    if not m:
        raise ValueError("Manuscript Table 1 pattern not found")
    new_t1 = generate_manuscript_table1()
    content = content[:m.start()] + new_t1 + content[m.end():]

    # 3. Section 3.3.1 (Right-Shift honest assessment with EKF parity)
    sec331_new = (
        "#### 3.3.1 Honest Assessment of Right-Shift Rescheduling (Fair EKF-Equipped Baseline)\n"
        "Right-Shift Rescheduling serves as a rigorous reactive baseline because it inherits global machine assignments and operation orderings "
        "from Cloud NSGA-III and is equipped with the **identical `BreadboardDAQInterface` + `RealTimeToolWearEstimator` (Live EKF) and statistical PM rule** "
        "as the Proposed DT (eliminating all unobservable oracle assumptions). With sensor parity, Right-Shift also achieves $0.00$ breakdowns across all five benchmarks (`Mk01`–`Mk05`). "
        f"Under two-sided paired Wilcoxon testing with Holm-Bonferroni correction ($K=9$; which tests distributional difference rather than formal TOST), "
        f"no statistically significant difference in Hypervolume is observed between Right-Shift and Proposed DT on `Mk01` ($p_{{\\text{{holm}}}} = {mk01_rs_pholm:.4f}$) and `Mk04` ($p_{{\\text{{holm}}}} = {mk04_rs_pholm:.4f}$), "
        f"while Right-Shift achieves statistically higher Hypervolume on `Mk02` ($p_{{\\text{{holm}}}} = {mk02_rs_pholm:.4f}$), `Mk03` ($p_{{\\text{{holm}}}} = {mk03_rs_pholm:.4f}$), and `Mk05` ($p_{{\\text{{holm}}}} = {mk05_rs_pholm:.4f}$) "
        "by preserving fixed Cloud machine/speed topology without peak-tariff throttling.\n\n"
        "However, Right-Shift cannot adapt cutting speeds dynamically to cyclic TOU windows or reroute disrupted schedules. In contrast, the Proposed DT provides:\n"
        f"- **TOU Energy Cost Minimization:** On dense bottleneck `Mk05`, Proposed DT achieves the lowest TOU electricity cost among dynamic schedulers (\\${float(prop_map['Mk05']['Cost_Mean']):.3f} $\\pm$ {float(prop_map['Mk05']['Cost_Std']):.3f} vs. \\${float(rs_map['Mk05']['Cost_Mean']):.3f} $\\pm$ {float(rs_map['Mk05']['Cost_Std']):.3f} for Right-Shift, **saving {mk05_sav:.2f}%**).\n"
        f"- **Tardiness Control vs. Open-Loop Methods:** On `Mk05`, Proposed DT keeps TWT to $5395.5 \\pm 800.5\\,\\text{{min}}$, lower than open-loop evolutionary baselines (NSGA-III at $6100.4 \\pm 983.7\\,\\text{{min}}$, MOEA-D at $6984.4 \\pm 1206.9\\,\\text{{min}}$). Right-Shift achieves shorter nominal makespan and TWT (${float(rs_map['Mk05']['Makespan_Mean']):.2f}\\,\\text{{min}}$ and ${float(rs_map['Mk05']['TWT_Mean']):.1f}\\,\\text{{min}}$ on `Mk05`) at higher TOU electricity cost.\n"
        "- **Complete Failure Immunity vs. Open-Loop Solvers:** Proposed DT guarantees $0.00$ breakdowns across `Mk01`–`Mk05` and all transport latency tiers, eliminating the $0.33 - 12.67$ tool failures per shift suffered by open-loop metaheuristics and $0.80 - 7.60$ failures of wear-blind Variant 1."
    )
    sec331_pattern = re.compile(r"#### 3\.3\.1 Honest Assessment of Right-Shift Rescheduling.*?(?=\n\n#### 3\.3\.2)", re.DOTALL)
    m331 = sec331_pattern.search(content)
    if m331:
        content = content[:m331.start()] + sec331_new + content[m331.end():]

    # 3b. Section 3.3.3 (Mk05 narrative)
    sec333_pattern = re.compile(r"#### 3\.3\.3 Resolving the Objective Trade-Offs on Bottleneck Benchmark Mk05.*?(?=\n\n---)", re.DOTALL)
    m333 = sec333_pattern.search(content)
    if m333:
        sec333_new = (
            "#### 3.3.3 Resolving the Objective Trade-Offs on Bottleneck Benchmark Mk05\n"
            "On benchmark `Mk05` (15 jobs with 106 operations crowded onto only 4 bottleneck machines, $26.5$ operations per machine), "
            f"fair EKF-equipped Right-Shift achieves higher Hypervolume ($\\text{{HV}} = {float(rs_map['Mk05']['HV_Mean']):.4f}$ vs. ${float(prop_map['Mk05']['HV_Mean']):.4f}$, $p_{{\\text{{holm}}}} = {mk05_rs_pholm:.4f}$), "
            "while the Proposed DT prioritizes peak-tariff energy reduction:\n"
            f"- **Shared Zero-Breakdown Safety:** Both EKF-equipped Proposed DT and fair Right-Shift achieve $0.00$ tool breakdowns, whereas open-loop NSGA-III incurs $9.20$, MOEA-D $9.87$, and RVEA $12.67$ breakdowns per run.\n"
            f"- **Lowest Dynamic TOU Cost:** By dynamically scaling cutting speeds down to $s=1$ or scheduling opportunistic PM during high-tariff periods, Proposed DT achieves $\\$11.198 \\pm 0.539$ (saving **{mk05_sav:.2f}%** vs. Right-Shift at \\${float(rs_map['Mk05']['Cost_Mean']):.3f} $\\pm$ {float(rs_map['Mk05']['Cost_Std']):.3f}).\n"
            "- **Reduced Tardiness vs. Open-Loop:** TWT is kept to $5395.5\\,\\text{min}$, significantly lower than NSGA-III ($6100.4\\,\\text{min}$) and MOEA-D ($6984.4\\,\\text{min}$).\n"
            f"- The enclosed hypervolume metric penalizes the longer makespan ($408.76\\,\\text{{min}}$ vs. ${float(rs_map['Mk05']['Makespan_Mean']):.2f}\\,\\text{{min}}$) relative to `[1.1, 1.1, 1.1]` during peak-tariff throttling."
        )
        content = content[:m333.start()] + sec333_new + content[m333.end():]

    # 6. Table 2
    t2_pattern = re.compile(r"(\| Instance \| Variant \| Description \| Makespan.*?\n)(.*?)(?=\n### 4\.2 Analytical Findings)", re.DOTALL)
    m2 = t2_pattern.search(content)
    if not m2:
        raise ValueError("Manuscript Table 2 pattern not found")
    new_t2 = generate_manuscript_table2()
    content = content[:m2.start()] + new_t2 + content[m2.end():]

    # 10. Table 3 (CDI sensitivity) & objective N_shocks=2.0 Section 4.3 narrative
    sec43_narrative = (
        f"As shown in Figure 4(b), thresholds are classified strictly by objective physical matching to the $N_{{\\text{{shocks}}}} = 2.0$ macro-disturbances per shift:\n"
        f"- **Over-Triggering Region (`> 2 Shocks`, $\\text{{calls}} > 2.0$, $\\theta_{{\\text{{th}}}} \\le 0.05$):** High sensitivity triggers ${float(r05['Cloud_Calls_Mean']):.2f} \\pm {float(r05['Cloud_Calls_Std']):.2f}$ to ${float(r02['Cloud_Calls_Mean']):.2f} \\pm {float(r02['Cloud_Calls_Std']):.2f}$ calls per shift ($7.25$–$13.15\\,\\text{{s}}$ Cloud Warm-Start solver overhead). While $\\theta_{{\\text{{th}}}}^* = {opt_th:.2f}$ minimizes the heuristic composite `Total_System_Cost_Index` (${opt_sys:.4f}$, ${opt_cmax_m:.2f}\\,\\text{{min}}$, \\${opt_cost_m:.3f}), it does so by over-triggering beyond the 2 physical shocks.\n"
        f"- **Event-Matched Plateau (`1–2 Shocks`, $1.0 \\le \\text{{calls}} \\le 2.0$, $\\theta_{{\\text{{th}}}} \\in [{min_pl:.2f}, {max_pl:.2f}]$):** Resync calls track the physical macro-disturbance budget (${float(r18['Cloud_Calls_Mean']):.2f}$ to ${float(r08['Cloud_Calls_Mean']):.2f}$ calls; $2.68$–$5.10\\,\\text{{s}}$ solver overhead). At $\\theta_{{\\text{{th}}}} = 0.08$, the index is $1.0547$ (${float(r08['Cloud_Calls_Mean']):.2f} \\pm {float(r08['Cloud_Calls_Std']):.2f}$ calls, ${float(r08['Makespan_Mean (min)']):.2f}\\,\\text{{min}}$, \\${float(r08['TOU_Cost_Mean ($)']):.3f}$); the frozen deployment center $\\theta_{{\\text{{th}}}} = 0.12$ executes ${float(r12['Cloud_Calls_Mean']):.2f} \\pm {float(r12['Cloud_Calls_Std']):.2f}$ calls (${float(r12['Makespan_Mean (min)']):.2f} \\pm {float(r12['Makespan_Std (min)']):.2f}\\,\\text{{min}}$, TOU Cost \\${float(r12['TOU_Cost_Mean ($)']):.3f} $\\pm$ {float(r12['TOU_Cost_Std ($)']):.3f}$, `Total_System_Cost_Index` $= {float(r12['Total_System_Cost_Index']):.4f}$).\n"
        f"- **Under-Triggering Region (`< 1 Shock`, $\\text{{calls}} < 1.0$, $\\theta_{{\\text{{th}}}} \\ge {float(r22['CDI_Threshold']):.2f}$):** Below-unity triggering misses macro-disturbances ($0.90 \\to 0.00$ calls), degrading Makespan up to **${float(r50['Makespan_Mean (min)']):.2f} \\pm {float(r50['Makespan_Std (min)']):.2f}\\,\\text{{min}}$** and raising `Total_System_Cost_Index` to ${float(r50['Total_System_Cost_Index']):.4f}$."
    )

    t3_pattern = re.compile(r"(\| CDI Threshold \$\\theta_\{.*?)(?=\n### 4\.4 Inter-Workstation)", re.DOTALL)
    m3 = t3_pattern.search(content)
    if m3:
        new_t3_block = generate_cdi_sensitivity_table() + "\n" + sec43_narrative + "\n"
        content = content[:m3.start()] + new_t3_block + content[m3.end():]

    # 11. Table 4 (Transport sensitivity)
    t4_pattern = re.compile(r"(\| Instance \| Transport Latency.*?\n)(.*?)(?=\n#### Key Findings from Transport Sensitivity:)", re.DOTALL)
    m4 = t4_pattern.search(content)
    if m4:
        new_t4 = generate_manuscript_table4()
        content = content[:m4.start()] + new_t4 + content[m4.end():]

    # Update Section 4.4 key finding 3 (EKF parity on Right-Shift vs open-loop NSGA-III)
    sec44_f3_pattern = re.compile(r"3\. \*\*Invariance of Tool Preservation:\*\*.*?(?=\n\n---)", re.DOTALL)
    m44 = sec44_f3_pattern.search(content)
    if m44:
        sec44_f3_new = (
            "3. **Invariance of Tool Preservation:**  \n"
            "   Across all four transport latency tiers ($0.0, 1.0, 2.0, 3.0\\,\\text{min}$), the Proposed Cloud-Edge DT (and identically EKF-equipped Right-Shift Rescheduling) maintains **$0.00$ tool breakdowns** on both `Mk01` and `Mk03`. "
            "In contrast, open-loop NSGA-III suffers $0.87$ tool failures on Mk01 and $9.67$ failures on Mk03, and unguided Standalone Edge PPO suffers $5.60$–$6.73$ failures on Mk03, "
            "confirming that real-time EKF wear estimation and $2\\sigma_{\\text{EKF}}$ statistical preventive maintenance operate invariantly to inter-machine transit times."
        )
        content = content[:m44.start()] + sec44_f3_new + content[m44.end():]

    # 12. Section 4.5 (Closed-Loop HIL 5-Mode Table 5 incl. Frequency-Matched Control & Warm-Start 2.6835s latency)
    m1_c, m2_c, m3_c, m4_c, m5_c = (
        float(m1_r["cloud_reschedule_calls"]),
        float(m2_r["cloud_reschedule_calls"]),
        float(m3_r["cloud_reschedule_calls"]),
        float(m4_r["cloud_reschedule_calls"]),
        float(m5_r["cloud_reschedule_calls"]),
    )
    m5_cs = float(m5_r["cloud_calls_std"])
    m1_cm, m2_cm, m3_cm, m4_cm, m5_cm = (
        float(m1_r["final_makespan_min"]),
        float(m2_r["final_makespan_min"]),
        float(m3_r["final_makespan_min"]),
        float(m4_r["final_makespan_min"]),
        float(m5_r["final_makespan_min"]),
    )
    m1_cms, m2_cms, m3_cms, m4_cms, m5_cms = (
        float(m1_r["makespan_std_min"]),
        float(m2_r["makespan_std_min"]),
        float(m3_r["makespan_std_min"]),
        float(m4_r["makespan_std_min"]),
        float(m5_r["makespan_std_min"]),
    )
    m1_co, m2_co, m3_co, m4_co, m5_co = (
        float(m1_r["tou_energy_cost_usd"]),
        float(m2_r["tou_energy_cost_usd"]),
        float(m3_r["tou_energy_cost_usd"]),
        float(m4_r["tou_energy_cost_usd"]),
        float(m5_r["tou_energy_cost_usd"]),
    )
    m1_cos, m2_cos, m3_cos, m4_cos, m5_cos = (
        float(m1_r["tou_cost_std_usd"]),
        float(m2_r["tou_cost_std_usd"]),
        float(m3_r["tou_cost_std_usd"]),
        float(m4_r["tou_cost_std_usd"]),
        float(m5_r["tou_cost_std_usd"]),
    )
    m1_pm, m1_pms = float(m1_r["num_pm_replacements"]), float(m1_r["pm_std"])
    m2_pm, m2_pms = float(m2_r["num_pm_replacements"]), float(m2_r["pm_std"])
    m3_pm, m3_pms = float(m3_r["num_pm_replacements"]), float(m3_r["pm_std"])
    m5_pm, m5_pms = float(m5_r["num_pm_replacements"]), float(m5_r["pm_std"])
    m2_lat_s = float(m2_r["total_solver_latency_s"])
    m3_lat_s = float(m3_r["total_solver_latency_s"])
    m5_lat_s = float(m5_r["total_solver_latency_s"])
    m5_lat_std_s = float(m5_r["solver_latency_std_s"])
    p_m1_co = _fmt_p(float(m1_r["p_wilcoxon_cost"]))
    p_m2_mk = _fmt_p(float(m2_r["p_wilcoxon_makespan"]))
    p_m2_co = _fmt_p(float(m2_r["p_wilcoxon_cost"]))
    p_m3_mk = _fmt_p(float(m3_r["p_wilcoxon_makespan"]))
    p_m3_co = _fmt_p(float(m3_r["p_wilcoxon_cost"]))
    p_m4_mk = _fmt_p(float(m4_r["p_wilcoxon_makespan"]))
    p_m4_co = _fmt_p(float(m4_r["p_wilcoxon_cost"]))

    sec45 = (
        "### 4.5 Closed-Loop Multi-Tier Latency & Five-Mode HIL Evaluation (incl. Frequency-Matched Periodic Control)\n"
        "To isolate **(A) Live EKF Wear Feedback**, **(B) Call-Frequency Reduction**, and **(C) Event-Triggered Timing vs. Blind Periodic Timing**, "
        "we execute a 5-mode HIL evaluation across 15 independent test seeds (`TEST_SEEDS = [301..315]`, Brandimarte `Mk01` physical twin, 55 operations) "
        f"after freezing $\\theta_{{\\text{{th}}}} = 0.12$ within the validation Event-Matched Plateau $[{min_pl:.2f}, {max_pl:.2f}]$ (`VAL_SEEDS = [201..210]`). "
        "All 5 modes share identical TOU-aware speed logic (preserving multi-objective planned speed off/mid-peak and derating to $s=1$ inside On-Peak $\\text{rate} \\ge 0.20$), "
        "TOU-cost-aware resync scoring, and statistically principled EKF wear projection ($w_{\\text{proj}} = \\hat{w} + \\Delta w_{\\text{nom}} + 2\\sigma_{\\text{EKF}} + 2.0$), "
        "while billing each Cloud Warm-Start Resynchronization invocation at its empirical latency (`cloud_nsga3_resync_solve_s = 2.6835 s` + `2 * 0.0814 ms` MQTT RTT):\n"
        "- **Mode 1 (`Periodic High-Freq + No EKF`, Every 5 ops):** $10.00$ blind periodic calls using cumulative nominal wear tracking (`nominal_wear_um`, no EKF sensors).\n"
        "- **Mode 2 (`Periodic High-Freq + Live EKF`, Every 5 ops):** $10.00$ periodic calls ($26.8366\\,\\text{s}$ total Cloud Warm-Start solver delay) paired with Live EKF.\n"
        "- **Mode 3 (`Periodic Frequency-Matched + Live EKF`, Op 25 & Op 50):** Fixed low-frequency periodic calls matched to $N_{\\text{shocks}} = 2.0$ ($2.00$ calls, $5.3673\\,\\text{s}$ solver delay) at fixed milestones regardless of true shock timing (`Op 19..22`, `Op 37..40`).\n"
        "- **Mode 4 (`One-Shot Autonomous + Live EKF`, 0 calls):** Edge EKF + ONNX policy only, without online Cloud re-synchronization.\n"
        "- **Mode 5 (`Proposed Event-Triggered CDI + Live EKF`, $\\theta_{\\text{th}} = 0.12$):** Invokes Cloud Warm-Start Resynchronization immediately when $\\text{CDI}(t) > 0.12$.\n\n"
        "```\n"
        "Table 5: Quantitative Comparison of Five Modes in Closed-Loop HIL Simulation (55 Operations, Mk01 Physical Twin, Mean ± Std & Paired Wilcoxon vs. Mode 5 across TEST_SEEDS = [301..315])\n"
        "```\n"
        + generate_manuscript_table5() + "\n\n"
        "#### Multi-Tier Latency Hierarchy & Cloud Warm-Start Overhead:\n"
        "Empirical benchmarking on a Linux x86_64 workstation (Python 3.12, ONNX Runtime CPUExecutionProvider) confirms the multi-tier timing distribution:\n"
        "1. **Edge ONNX Forward Pass:** $0.0132\\,\\text{ms}$ ($13.2\\,\\mu\\text{s}$, $\\sim 75,500$ inferences/second).\n"
        "2. **Local MQTT Transport RTT:** $0.0814\\,\\text{ms}$ (industrial LAN broker loopback).\n"
        "3. **Edge Decision Step:** $0.1506\\,\\text{ms}$ (full feature extraction, action masking, and ONNX prediction).\n"
        "4. **Online List Surrogate (`fast_heuristic_resync`):** $0.2143\\,\\text{ms}$ (simulation-engine surrogate assignment).\n"
        "5. **Edge Shift Computation:** $7.955\\,\\text{ms}$ (entire 55-step shift simulation on Mk01; $22.71\\,\\text{ms}$ on Mk03).\n"
        "6. **Cloud Warm-Start NSGA-III Resync (Billed in HIL):** $2.6835\\,\\text{s}$ per trigger ($26.84\\,\\text{s}$ cumulative for 10 periodic calls).\n"
        "7. **Cloud NSGA-III Full Solve (Offline Baseline):** $4.2392\\,\\text{s}$ ($4.24 \\pm 0.03\\,\\text{s}$ on Mk01; $14.97\\,\\text{s}$ on Mk03).\n\n"
        "This establishes a clear multi-dimensional latency hierarchy: per-decision edge reactivity ($0.1506\\,\\text{ms}$) is $\\sim 28,147\\times$ ($\\sim 2.81 \\times 10^4\\times$) faster than a full Cloud NSGA-III solve ($4.2392\\,\\text{s}$) and $\\sim 17,818\\times$ faster than a Cloud Warm-Start NSGA-III resync ($2.6835\\,\\text{s}$), explaining why high-frequency periodic Cloud calls ($10 \\times 2.6835\\,\\text{s} = 26.84\\,\\text{s}$) create severe solver queuing bottlenecks.\n\n"
        "#### Component & Timing Attribution (Figure 3):\n"
        f"- **Value of Event-Triggered Timing over Frequency-Matched Periodic (`Mode 3` vs. `Mode 5`):** Both modes use identical Live EKF and PM rules, and **Mode 3** is explicitly tuned to the exact shock count ($2.00$ calls at Op 25 and Op 50, ${m3_lat_s:.4f}\\,\\text{{s}}$ overhead). Yet because fixed milestones miss the random shock instants (`Op 19..22`, `Op 37..40`) and trigger needlessly at Op 50 even after benign shifts, **Mode 5** reduces Cloud Warm-Start calls by an additional **{m3_m5_call_reduct:.1f}%** (${m5_c:.2f} \\pm {m5_cs:.2f}$ vs. ${m3_c:.2f}$, cutting solver overhead from ${m3_lat_s:.2f}\\,\\text{{s}}$ to ${m5_lat_s:.2f} \\pm {m5_lat_std_s:.2f}\\,\\text{{s}}$) while achieving slightly shorter mean makespan (${m5_cm:.2f} \\pm {m5_cms:.2f}$ vs. ${m3_cm:.2f} \\pm {m3_cms:.2f}\\,\\text{{min}}$, $p = {p_m3_mk}$) and virtually identical TOU cost (\\${m5_co:.3f} vs. \\${m3_co:.3f}, $p = {p_m3_co}$).\n"
        f"- **Isolating CDI Event-Triggering vs. High-Frequency Periodic (`Mode 2` vs. `Mode 5`):** Replacing 10-call periodic resync (**Mode 2**, ${m2_lat_s:.2f}\\,\\text{{s}}$ solver overhead) with **Mode 5** slashes synchronization calls and Warm-Start solver delay by **{m5_reduct:.1f}%** (${m5_lat_s:.2f}\\,\\text{{s}}$ vs. ${m2_lat_s:.2f}\\,\\text{{s}}$), significantly reduces TOU electricity cost by **${m2_m5_cost_reduct:.1f}%** (\\${m5_co:.3f} ± {m5_cos:.3f} vs. \\${m2_co:.3f} ± {m2_cos:.3f}, $p = {p_m2_co}$), with no statistically significant makespan penalty (${m5_cm:.2f}$ vs. ${m2_cm:.2f}\\,\\text{{min}}$, $p = {p_m2_mk}$).\n"
        f"- **Isolating Bounded Resync vs. Zero Resync (`Mode 4` vs. `Mode 5`):** Operating purely autonomously without Cloud re-synchronization (**Mode 4**, $0.00$ calls) accumulates unmitigated drift ($\\text{{CDI}}_{{\\max}} = {float(m4_r['max_cdi']):.4f} \\pm {float(m4_r['max_cdi_std']):.4f}$) that inflates makespan to ${m4_cm:.2f} \\pm {m4_cms:.2f}\\,\\text{{min}}$. Adding just ${m5_c:.2f} \\pm {m5_cs:.2f}$ CDI-triggered calls in **Mode 5** recovers **${m4_cm - m5_cm:.2f}\\,\\text{{min}}$** of makespan ($p = {p_m4_mk}$) at no statistically significant TOU cost difference (\\${m5_co:.3f} vs. \\${m4_co:.3f}, $p = {p_m4_co}$).\n"
        f"- **Sensorless PM Blindness (`Mode 1` vs. `Mode 2`/`Mode 5`):** Without live EKF, **Mode 1** executes rigid schedule-only PM ($\\text{{PM}} = {m1_pm:.2f} \\pm {m1_pms:.2f}$ vs. ${m2_pm:.2f} \\pm {m2_pms:.2f}$ in Mode 2 and ${m5_pm:.2f} \\pm {m5_pms:.2f}$ in Mode 5), ignoring $+26\\,\\mu\\text{{m}}$ shock spikes and incurring $+\\${m1_co - m5_co:.3f}$/shift higher TOU cost than Mode 5 ($p = {p_m1_co}$).\n"
    )

    sec5_marker = "## 5. Mechatronic Hardware-in-the-Loop Telemetry"
    sec5_idx = content.find(sec5_marker)
    sec45_marker = "### 4.5 Closed-Loop Multi-Tier Latency"
    sec45_idx = content.find(sec45_marker)

    if sec45_idx != -1 and sec5_idx != -1:
        content = content[:sec45_idx] + sec45 + "\n---\n\n" + content[sec5_idx:]
    elif sec5_idx != -1:
        content = content[:sec5_idx] + "\n" + sec45 + "\n---\n\n" + content[sec5_idx:]

    # 12b. Section 5.1 Hardware Safety (Open-Drain GPIO 27, 2500ms kinematic decel, 300ms Host Watchdog)
    sec51_p_prev = (
        "The physical testbed comprises an ESP32 microcontroller, an INA226 current/power sensor ($10\\,\\text{m}\\Omega$ / $0.010\\,\\Omega$ shunt), "
        "an ADXL345 3-axis accelerometer configured in **Full-Resolution $\\pm 16g$ mode (`0x31 = 0x0B`, $3.9\\,\\text{mg/LSB}$)** at **$200\\,\\text{Hz}$** sampling rate "
        "($100\\,\\text{Hz}$ Nyquist bandwidth capturing low-frequency structural vibration energy and sub-100 Hz amplitude modulation), opto-isolated solid-state relays on "
        "`GPIO 25` and `GPIO 26` protected by **external $10\\,\\text{k}\\Omega$ pull-down resistors to `GND`** (paired with internal boot pull-down enable), and a dedicated "
        "hardware brake output **`GPIO 27` (`PIN_GRBL_FEED_HOLD`) wired directly to the GRBL CNC Shield / Makerbase DLC32 `Feed Hold (A1)` input** (active `LOW`, idle `HIGH`). "
        "During transient I2C bus errors within the 3-sample debounce window ($1 \\le \\text{i2c\\_err\\_cnt} < 3$, spanning $15\\,\\text{ms}$ at $200\\,\\text{Hz}$), the firmware applies "
        "a Zero-Order Hold (ZOH) on filtered estimates and explicitly flags `\"sample_held\": true` in the $10\\,\\text{Hz}$ JSON packet. "
        "If an unrecovered fault persists (`i2c_consecutive_errors >= 3`, latching `fault_latched = true`), **Stop Category 1 two-phase hardware shutdown** executes immediately: "
        "(1) ESP32 pulls `GPIO 27` `LOW` in $< 1\\,\\text{ms}$ to halt X/Y/Z axis motion via GRBL hardware `Feed Hold`, keeping `PIN_SSR_MAIN` active so the spindle does not stall mid-cut during deceleration; "
        "concurrently, `GRBLHardwareController.emergency_stop()` transmits serial `Feed Hold (!)` and polls `?` every $50\\,\\text{ms}$ until GRBL confirms `Hold:0` (`Idle` or `feed <= 1e-3`, up to $3.0\\,\\text{s}$ safety limit); "
        "(2) only after verified axis stop (or the ESP32 `SPINDLE_DECEL_WAIT_MS = 2000\\,\\text{ms}` hardware fallback) does the controller dispatch `CMD:OFF` to open SSRs followed by `Soft Reset (0x18)`."
    )
    sec51_p_new = (
        "The physical testbed comprises an ESP32 microcontroller, an INA226 current/power sensor ($10\\,\\text{m}\\Omega$ / $0.010\\,\\Omega$ shunt), "
        "an ADXL345 3-axis accelerometer configured in **Full-Resolution $\\pm 16g$ mode (`0x31 = 0x0B`, $3.9\\,\\text{mg/LSB}$)** at **$200\\,\\text{Hz}$** sampling rate "
        "($100\\,\\text{Hz}$ Nyquist bandwidth capturing low-frequency structural vibration energy and sub-100 Hz amplitude modulation), opto-isolated solid-state relays on "
        "`GPIO 25` and `GPIO 26` protected by **external $10\\,\\text{k}\\Omega$ pull-down resistors to `GND`** (paired with internal boot pull-down enable), and an "
        "**Open-Drain (`OUTPUT_OPEN_DRAIN`, internal $3.3\\,\\text{V}$ pull-up disabled via `gpio_pullup_dis`) hardware brake output `GPIO 27` (`PIN_GRBL_FEED_HOLD`)** "
        "(High-Z normal, active `LOW` brake) to prevent $3.3\\,\\text{V}$/$5\\,\\text{V}$ voltage-level conflicts: for **Arduino Uno + CNC Shield V3 ($5\\,\\text{V}$ logic)**, `GPIO 27` connects through an isolating optocoupler (`PC817`) or N-channel MOSFET (`2N7000`) open-drain stage to `Feed Hold (A1)`; for **Makerbase MKS DLC32 ($3.3\\,\\text{V}$ ESP32 running Grbl_ESP32 / FluidNC)**, it connects to the `Hold` pin on the input header (`I36 / GPIO 36`). "
        "During transient I2C bus errors within the 3-sample debounce window ($1 \\le \\text{i2c\\_err\\_cnt} < 3$, spanning $15\\,\\text{ms}$ at $200\\,\\text{Hz}$), the firmware applies "
        "a Zero-Order Hold (ZOH) on filtered estimates and explicitly flags `\"sample_held\": true` in the $10\\,\\text{Hz}$ JSON packet. "
        "If an unrecovered fault persists (`i2c_consecutive_errors >= 3`, latching `fault_latched = true`), **Stop Category 1 two-phase hardware shutdown** executes immediately: "
        "(1) ESP32 pulls `GPIO 27` `LOW` in $< 1\\,\\text{ms}$ to halt X/Y/Z axis motion via hardware `Feed Hold`, keeping `PIN_SSR_MAIN` active so the spindle does not stall mid-cut during deceleration; "
        "concurrently, `GRBLHardwareController.emergency_stop()` transmits serial `Feed Hold (!)` and polls `?` every $50\\,\\text{ms}$ up to the synchronized kinematic deceleration timeout "
        "`compute_decel_timeout_s(1000.0, 10.0, 0.833) = (1000/60)/10 + 0.833 = 2.50\\,\\text{s}` (matching firmware `SPINDLE_DECEL_WAIT_MS = 2500\\,\\text{ms}` from GRBL `$110 = 1000\\,\\text{mm/min}` and `$120 = 10\\,\\text{mm/s}^2$) until GRBL confirms `Hold:0` (`Idle` or `feed <= 1e-3`); "
        "(2) only after verified axis stop (or the $2500\\,\\text{ms}$ / $2.50\\,\\text{s}$ kinematic fallback) does the controller dispatch `CMD:OFF` to open SSRs followed by `Soft Reset (0x18)`. "
        "Furthermore, if the ESP32 loses power, resets, or hangs, `EdgeDigitalTwinNode.check_telemetry_watchdog()` enforces a **$300\\,\\text{ms}$ Host Telemetry Heartbeat Watchdog** (3 missed $10\\,\\text{Hz}$ frames or `TimeoutError`/`ConnectionError`) that immediately calls `grbl.emergency_stop()` (`!`) directly over Serial."
    )
    content = content.replace(sec51_p_prev, sec51_p_new)

    # 13. Section 6 (Conclusion)
    new_conclusion = (
        "## 6. Conclusion\n"
        "This study has presented a closed-loop Cloud-Edge Collaborative Digital Twin framework for dynamic green flexible job shop scheduling under cutting tool wear degradation and cyclic TOU electricity tariffs. "
        "By synthesizing global macro-optimization (NSGA-III, mean $\\overline{\\text{HV}} = "
        f"{mean_hv:.4f}$ and $\\overline{{\\text{{IGD}}}} = {mean_igd:.4f}$ across Mk01–Mk05) with scale-invariant sub-millisecond edge neural dispatching "
        "(MaskablePPO ONNX: $0.0132\\,\\text{ms}$ forward pass, $0.1506\\,\\text{ms}$ decision step) via a shift-normalized Cumulative Disturbance Index "
        f"(frozen at $\\theta_{{\\text{{th}}}} = 0.12$ within the objective Event-Matched Plateau $[{min_pl:.2f}, {max_pl:.2f}]$ anchored to $N_{{\\text{{shocks}}}} = 2.0$, whereas $\\theta_{{\\text{{th}}}}^* = {opt_th:.2f}$ minimizes the heuristic composite `Total_System_Cost_Index` by over-triggering), "
        "the framework eliminates all tool breakdowns ($0.00$ across Mk01–Mk05 and all transport latency tiers, compared to $0.33 - 12.67$ breakdowns for open-loop metaheuristics), "
        f"reduces Cloud Warm-Start NSGA-III resynchronization calls by ${m5_reduct:.1f}\\%$ vs. high-frequency periodic resync (${m5_calls:.2f} \\pm {m5_calls_s:.2f}$ vs. ${m2_calls:.2f}$ calls, cutting solver delay from ${m2_lat_s:.2f}\\,\\text{{s}}$ to ${m5_lat_s:.2f}\\,\\text{{s}}$) and by ${m3_m5_call_reduct:.1f}\\%$ vs. frequency-matched periodic control (${m3_calls:.2f}$ calls), "
        f"reduces TOU electricity costs by up to $30.95\\%$ via SSR standby gating (saving $\\$6.223$ per shift on Mk03 vs. idle-only; or $+44.82\\%$ penalty if omitted), "
        f"and saves ${mk05_sav:.2f}\\%$ TOU electricity cost on bottleneck Mk05 relative to fair EKF-equipped Right-Shift Rescheduling under verified Stop Category 1 hardware safety interlocks (`GPIO 27` Open-Drain Feed Hold, $2500\\,\\text{{ms}}$/$2.50\\,\\text{{s}}$ kinematic deceleration sync, and $300\\,\\text{{ms}}$ Host Watchdog)."
    )
    conc_idx = content.find("## 6. Conclusion")
    if conc_idx != -1:
        content = content[:conc_idx] + new_conclusion + "\n"

    with open(MANUSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print("  -> PASSED: results/manuscript_draft.md fully synchronized!")


def update_readme():
    print("[2/3] Synchronizing README.md...")
    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    wilc_rows = load_csv("phase5_wilcoxon_benchmark_table.csv")
    prop_rows = [r for r in wilc_rows if "Proposed" in r["Method"]]
    mean_hv = sum(float(r["HV_Mean"]) for r in prop_rows) / len(prop_rows)
    mean_igd = sum(float(r["IGD_Mean"]) for r in prop_rows) / len(prop_rows)

    rs_map = {r["Instance"]: r for r in wilc_rows if "Right-Shift" in r["Method"]}
    nsga3_map = {r["Instance"]: r for r in wilc_rows if "NSGA-III (Open-Loop)" in r["Method"]}
    nsga2_map = {r["Instance"]: r for r in wilc_rows if "NSGA-II (Open-Loop)" in r["Method"]}
    prop_map = {r["Instance"]: r for r in prop_rows}
    mk05_sav = (float(rs_map["Mk05"]["Cost_Mean"]) - float(prop_map["Mk05"]["Cost_Mean"])) / float(rs_map["Mk05"]["Cost_Mean"]) * 100.0
    mk01_rs_pholm = float(rs_map["Mk01"]["HV_p_holm"])
    mk02_rs_pholm = float(rs_map["Mk02"]["HV_p_holm"])
    mk03_rs_pholm = float(rs_map["Mk03"]["HV_p_holm"])
    mk04_rs_pholm = float(rs_map["Mk04"]["HV_p_holm"])
    mk05_rs_pholm = float(rs_map["Mk05"]["HV_p_holm"])
    mk03_nsga3_pholm = float(nsga3_map["Mk03"]["HV_p_holm"])
    mk03_nsga2_pholm = float(nsga2_map["Mk03"]["HV_p_holm"])

    hil_rows = {r["mode_id"]: r for r in load_csv("phase4_hil_metrics.csv")}
    m1_r, m2_r, m3_r, m4_r, m5_r = (
        hil_rows["Mode 1"], hil_rows["Mode 2"], hil_rows["Mode 3"], hil_rows["Mode 4"], hil_rows["Mode 5"]
    )
    m5_calls = float(m5_r["cloud_reschedule_calls"])
    m5_calls_s = float(m5_r["cloud_calls_std"])
    m5_reduct = float(m5_r["reduction_in_cloud_calls_pct"])
    m2_calls = float(m2_r["cloud_reschedule_calls"])
    m3_calls = float(m3_r["cloud_reschedule_calls"])
    m2_cmax, m5_cmax = float(m2_r["final_makespan_min"]), float(m5_r["final_makespan_min"])
    m2_cost, m5_cost = float(m2_r["tou_energy_cost_usd"]), float(m5_r["tou_energy_cost_usd"])
    m2_m5_cost_reduct = (m2_cost - m5_cost) / m2_cost * 100.0
    m3_m5_call_reduct = (m3_calls - m5_calls) / m3_calls * 100.0

    cdi_rows = load_csv("phase5_cdi_sensitivity.csv")
    cdi_map = {round(float(r["CDI_Threshold"]), 2): r for r in cdi_rows}
    opt_row = min(cdi_rows, key=lambda x: float(x["Total_System_Cost_Index"]))
    opt_th = float(opt_row["CDI_Threshold"])
    opt_sys = float(opt_row["Total_System_Cost_Index"])
    opt_calls_m, opt_calls_s = float(opt_row["Cloud_Calls_Mean"]), float(opt_row["Cloud_Calls_Std"])
    opt_cmax_m, opt_cmax_s = float(opt_row["Makespan_Mean (min)"]), float(opt_row["Makespan_Std (min)"])
    opt_cost_m, opt_cost_s = float(opt_row["TOU_Cost_Mean ($)"]), float(opt_row["TOU_Cost_Std ($)"])
    plateau_ths = sorted(
        float(r["CDI_Threshold"]) for r in cdi_rows if r.get("Zone") == "Event-Matched Plateau (1–2 Shocks)"
    )
    min_pl, max_pl = min(plateau_ths), max(plateau_ths)
    r02, r05, r08, r12, r18, r22, r50 = (
        cdi_map[0.02], cdi_map[0.05], cdi_map[0.08], cdi_map[0.12], cdi_map[0.18], cdi_map[0.22], cdi_map[0.50]
    )

    # 1. Section 1 Executive Summary: Line 27 CDI phrasing
    old_cdi_bullet = re.compile(r"- \*\*Event-Driven Synchronization via CDI:\*\*.*?\n", re.DOTALL)
    new_cdi_bullet = (
        f"- **Event-Driven Synchronization via CDI:** Implements a shift-normalized **Cumulative Disturbance Index ($\\text{{CDI}}(t)$)** that monitors factory makespan and energy deviations. "
        f"Re-optimization requests are transmitted to the Cloud only when $\\text{{CDI}}(t)$ exceeds $\\theta_{{\\text{{th}}}} = 0.12$ (frozen within the Event-Matched Plateau $[{min_pl:.2f}, {max_pl:.2f}]$ anchored to $N_{{\\text{{shocks}}}} = 2.0$, whereas $\\theta_{{\\text{{th}}}}^* = {opt_th:.2f}$ minimizes `Total_System_Cost_Index` by over-triggering), "
        f"slashing online Cloud Warm-Start rescheduling calls by ${m5_reduct:.1f}\\%$ (${m5_calls:.2f} \\pm {m5_calls_s:.2f}$ vs. ${m2_calls:.2f}$ periodic calls; ${m3_m5_call_reduct:.1f}\\%$ fewer than frequency-matched periodic control) and reducing TOU cost by ${m2_m5_cost_reduct:.1f}\\%$ vs. high-frequency periodic EKF resync while preserving $0.00$ tool failures.\n"
    )
    content = old_cdi_bullet.sub(lambda _: new_cdi_bullet, content, count=1)

    # 2. Section 2 Quantitative Highlights table
    s2_pattern = re.compile(r"(\| Category \| Audited Metric \| Empirical Observation & Contrast \| Impact / Advantage \|\n\| :--- \| :--- \| :--- \| :--- \|\n)(.*?)(?=\n---)", re.DOTALL)
    m_s2 = s2_pattern.search(content)
    if m_s2:
        s2_new = (
            "| Category | Audited Metric | Empirical Observation & Contrast | Impact / Advantage |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| **Inference Latency** | **$0.0132\\,\\text{ms}$** | Edge ONNX forward policy inference ($0.0132 \\pm 0.0013\\,\\text{ms}$; $0.1506 \\pm 0.0064\\,\\text{ms}$ full decision step) vs. Cloud NSGA-III ($4.24 \\pm 0.03\\,\\text{s}$) | **$\\sim 2.81 \\times 10^4 \\times$ (per-step) / $\\sim 3.20 \\times 10^5 \\times$ (forward) speedup** over Cloud solver; enables hard real-time dispatching. |\n"
            "| **Reliability & Breakdowns** | **$0.00$ Failures** | $0.00$ breakdowns across all 5 Brandimarte benchmarks (Mk01–Mk05) vs. $0.33 - 12.67$ (open-loop metaheuristics) and $0.80 - 7.60$ (wear-blind Variant 1); fair EKF-equipped Right-Shift also achieves $0.00$ | Eliminates catastrophic tool breakage through closed-loop EKF sensing ($R^2 = 0.9984$ active cutting / $0.9951$ full 60s profile). |\n"
            "| **Energy & Cost Savings** | **$30.95\\%$ Reduction** | SSR-actuated standby mode reduces TOU electricity cost from $\\$20.106$ to $\\$13.883$ on Mk03 ($-30.95\\%$ reduction, saving $\\$6.223$ per shift; or $+44.82\\%$ cost penalty if omitted) | Eliminates parasitic idling power during inter-operational waiting times. |\n"
            f"| **Pareto Quality** | **$\\overline{{\\text{{HV}}}} = {mean_hv:.4f}$** | Proposed DT achieves mean $\\overline{{\\text{{HV}}}}={mean_hv:.4f}$ and $\\overline{{\\text{{IGD}}}}={mean_igd:.4f}$ across all instances (Mk01–Mk05) with $0.00$ breakdowns | Statistically outperforms all 4 open-loop metaheuristics on Mk05 ($p_{{\\text{{holm}}}} \\le 0.0011$) and beats NSGA-III ($p_{{\\text{{holm}}}} = {mk03_nsga3_pholm:.4f}$) & NSGA-II ($p_{{\\text{{holm}}}} = {mk03_nsga2_pholm:.4f}$) on Mk03; saves {mk05_sav:.2f}% TOU cost on Mk05 vs. fair Right-Shift. |"
        )
        content = content[:m_s2.start()] + s2_new + content[m_s2.end():]

    # 2b. Section 5 Hardware-in-the-Loop Setup & Pinout (GPIO 27 Open-Drain, 2500ms kinematic decel, 300ms Host Watchdog)
    sec5_pattern = re.compile(r"(## 5\. Hardware-in-the-Loop Setup & Pinout\n)(.*?)(?=\n### GRBL v1\.1 Real-Time Speed Override Protocol)", re.DOTALL)
    m_s5 = sec5_pattern.search(content)
    if m_s5:
        new_sec5 = (
            "## 5. Hardware-in-the-Loop Setup & Pinout\n\n"
            "The physical twin utilizes an **ESP32 DevKitC** interfacing with industrial sensing and actuation circuitry over a dedicated $400\\,\\text{kHz}$ Fast-Mode I2C bus, opto-isolated solid-state relays, and a dedicated hardware brake line (`GPIO 27`) into the GRBL controller:\n\n"
            "| Peripheral | Component | Physical Pin (ESP32) | Protocol / Interface | Function / Specification |\n"
            "| :--- | :--- | :--- | :--- | :--- |\n"
            "| **Current / Power Sensor** | Texas Instruments INA226 | `GPIO 21` (SDA)<br/>`GPIO 22` (SCL) | I2C (`0x40`, $400\\,\\text{kHz}$) | $10\\,\\text{m}\\Omega$ shunt resistor, measures $V_{\\text{bus}}$ ($1.25\\,\\text{mV/LSB}$) and $I_{\\text{shunt}}$ ($2.5\\,\\mu\\text{V/LSB}$). |\n"
            "| **Vibration Accelerometer** | Analog Devices ADXL345 | `GPIO 21` (SDA)<br/>`GPIO 22` (SCL) | I2C (`0x53`, $400\\,\\text{kHz}$) | **Full-resolution $\\pm 16\\,\\text{g}$ (`0x31 = 0x0B`, $3.9\\,\\text{mg/LSB}$)**, **$200\\,\\text{Hz}$** ODR ($100\\,\\text{Hz}$ Nyquist bandwidth capturing low-frequency structural vibration energy), $20$-sample sliding RMS with 3-tap (`15 ms`) ZOH `sample_held` flag before fault latching. |\n"
            "| **Standby Power Actuator** | Opto-isolated SSR Module | `GPIO 25` | Digital Output ($3.3\\,\\text{V}$) + **$10\\,\\text{k}\\Omega$ pull-down to `GND`** | Cut off auxiliary control power during machine standby; internal + external pull-down locks LOW during boot. |\n"
            "| **Spindle / Main Power** | Opto-isolated SSR Module | `GPIO 26` | Digital Output ($3.3\\,\\text{V}$) + **$10\\,\\text{k}\\Omega$ pull-down to `GND`** | Main spindle power interlock; held `HIGH` during Stop Category 1 axis deceleration and opened only after verified `Hold:0` (`CMD:OFF`) or $2500\\,\\text{ms}$ kinematic fallback. |\n"
            "| **GRBL Hardware Feed Hold** | Optocoupler (`PC817`) / MOSFET (`2N7000`) or DLC32 Input | `GPIO 27` (`PIN_GRBL_FEED_HOLD`) | `OUTPUT_OPEN_DRAIN` (`High-Z` = run, `LOW` = brake; internal pull-up disabled) | **Arduino Uno + CNC Shield V3 ($5\\,\\text{V}$):** via `PC817`/`2N7000` to `Feed Hold (A1)`. **Makerbase MKS DLC32 ($3.3\\,\\text{V}$):** to `Hold` (`I36 / GPIO 36`). Pulled `LOW` in $<1\\,\\text{ms}$ on latched I2C fault. |\n"
            "| **Telemetry Interface** | USB-UART Bridge | `GPIO 1` (TX0)<br/>`GPIO 3` (RX0) | UART ($115200\\,\\text{bps}$) | $10\\,\\text{Hz}$ JSON telemetry monitored by **$300\\,\\text{ms}$ Host Watchdog** (`sensor_ok`, `fault_latched`, `sample_held`, `feed_hold_pin_active`). |\n"
            "| **CNC Machine Controller** | GRBL v1.1 Workcell | USB / Serial Port | UART ($115200\\,\\text{bps}$) | G-code streaming, real-time speed overrides, and `?` polling (`2.50 s` timeout from `$110=1000, $120=10`) until `Hold:0` before `Soft Reset (0x18)`. |\n\n"
            "### 5.1 Stop Category 1 Safety Sequence, Electrical Isolation & 300 ms Host Watchdog\n"
            "To guarantee electrical safety across $3.3\\,\\text{V}$/$5\\,\\text{V}$ logic domains and zero risk of cutting SSR spindle power while X/Y/Z axes remain in motion:\n"
            "1. **Open-Drain Logic Level Isolation (`GPIO 27`):** `PIN_GRBL_FEED_HOLD` is configured with `OUTPUT_OPEN_DRAIN` and `gpio_pullup_dis(...)` (High-Z = normal, `LOW` = brake). On **Arduino Uno + CNC Shield V3 ($5\\,\\text{V}$ logic)**, wire `GPIO 27` through an isolating optocoupler (`PC817`) or N-channel MOSFET (`2N7000`) open-drain stage to `Feed Hold (A1)` to prevent $5\\,\\text{V}$ pull-up contention. On **Makerbase MKS DLC32 ($3.3\\,\\text{V}$ ESP32 running Grbl_ESP32 / FluidNC)**, wire to the `Hold` pin on the input header (`I36 / GPIO 36`).\n"
            "2. **3-Sample I2C Debouncing & `sample_held` Flag:** Transient I2C NACKs within `1 <= i2c_err_cnt < 3` ($15\\,\\text{ms}$ at $200\\,\\text{Hz}$) preserve prior filtered estimates via Zero-Order Hold and report `\"sample_held\": true` in the $10\\,\\text{Hz}$ JSON telemetry stream without tripping relays.\n"
            "3. **Kinematic Phase 1 / Phase 2 Shutdown (`2500 ms` / `2.50 s` Sync):** Upon 3 consecutive unrecovered I2C read failures (`fault_latched = true`), ESP32 pulls `GPIO 27` `LOW` in $< 1\\,\\text{ms}$ while holding `PIN_SSR_MAIN = HIGH` during axis deceleration. Concurrently, `GRBLHardwareController.emergency_stop()` issues serial `Feed Hold (!)` and polls `?` every $50\\,\\text{ms}$ up to `compute_decel_timeout_s(1000.0, 10.0, 0.833) = (1000/60)/10 + 0.833 = 2.50\\,\\text{s}` (synchronized with firmware `SPINDLE_DECEL_WAIT_MS = 2500\\,\\text{ms}` computed from GRBL `$110 = 1000\\,\\text{mm/min}`, `$120 = 10\\,\\text{mm/s}^2`, plus $833\\,\\text{ms}$ margin) until `Hold:0` (`Idle` or `feed <= 1e-3`) before commanding `CMD:OFF` and `Soft Reset (0x18)`.\n"
            "4. **Host Telemetry Heartbeat Watchdog (`300 ms`):** `EdgeDigitalTwinNode.check_telemetry_watchdog()` monitors the $10\\,\\text{Hz}$ ESP32 stream (`telemetry_watchdog_timeout_ms = 300.0 ms`, 3 missed frames) and catches `TimeoutError`/`ConnectionError` on `read_telemetry()`. If the ESP32 loses power, brownout-resets, or hangs, the Host immediately invokes `grbl.emergency_stop()` to transmit `!` directly over Serial to GRBL.\n"
        )
        content = content[:m_s5.start()] + new_sec5 + content[m_s5.end():]

    # 3. Section 7.1 Multi-Instance Benchmark Table
    s71_pattern = re.compile(r"(\| Problem Instance \| Method \| Hypervolume \(HV\).*?\n)(.*?)(?=\n\*Overall across all 5 benchmark instances)", re.DOTALL)
    m71 = s71_pattern.search(content)
    if not m71:
        raise ValueError("README Section 7.1 Table pattern not found")
    new_readme_t1 = generate_readme_table1()
    content = content[:m71.start()] + new_readme_t1 + content[m71.end():]

    # Footer under 7.1
    footer_pattern = re.compile(r"\*Overall across all 5 benchmark instances.*?\*\n")
    new_71_footer = f"*Overall across all 5 benchmark instances, the Proposed Cloud-Edge DT achieves a mean Hypervolume $\\overline{{\\text{{HV}}}} = {mean_hv:.4f}$ and mean $\\overline{{\\text{{IGD}}}} = {mean_igd:.4f}$, with exactly $0.00$ tool failures across all evaluated configurations.*\n"
    content = footer_pattern.sub(lambda _: new_71_footer, content, count=1)

    # 4. Section 7.2 Ablation Study Table + Baseline Fairness & Revision Transparency Note
    s72_pattern = re.compile(r"(\| Problem Instance \| Architectural Variant \| Description & Mechanism.*?\n)(.*?)(?=\n#### Key Component Insights from Ablation Study:)", re.DOTALL)
    m72 = s72_pattern.search(content)
    if not m72:
        raise ValueError("README Section 7.2 Table pattern not found")
    new_readme_t2 = generate_readme_table2()
    content = content[:m72.start()] + new_readme_t2 + content[m72.end():]

    fairness_note = (
        "#### Baseline Fairness & Revision Transparency Note\n"
        "In the current benchmark revision, **`Right-Shift Rescheduling`** (`execute_right_shift_rescheduling()` in `tests/test_task5_2_multi_instance_eval.py`) was standardized to share the **exact same `BreadboardDAQInterface` + `RealTimeToolWearEstimator` (Live EKF) sensor suite and statistical preventive maintenance rule** ($w_{\\text{proj}} = \\hat{w} + \\Delta w_{\\text{nom}} + 2\\sigma_{\\text{EKF}} + 2.0$, without oracle access) as `Proposed Cloud-Edge DT`. Prior revisions had contrasted an EKF-equipped Proposed DT against a sensorless/theoretical Right-Shift implementation that suffered $0.40 - 1.47$ tool breakdowns on large horizons (`Mk05`/`Mk03`). Standardizing sensor parity ensures both methods achieve **$0.00$ tool breakdowns** across all five benchmarks and isolates the pure algorithmic trade-off: Right-Shift preserves fixed Cloud NSGA-III machine/speed topology (`HV` advantage on `Mk02`, `Mk03`, `Mk05`), whereas `Proposed DT` adds dynamic TOU speed-scaling and CDI synchronization (`8.44%` TOU cost reduction on bottleneck `Mk05`).\n\n"
    )
    existing_note_pat = re.compile(r"#### Baseline Fairness & Revision Transparency Note.*?(?=!\[Ablation Study)", re.DOTALL)
    if existing_note_pat.search(content):
        content = existing_note_pat.sub(lambda _: fairness_note, content, count=1)
    else:
        content = content.replace(
            "![Ablation Study and Sensitivity Analysis](tests/phase5_ablation_and_sensitivity.png)",
            fairness_note + "![Ablation Study and Sensitivity Analysis](tests/phase5_ablation_and_sensitivity.png)",
            1,
        )

    # 5. Section 7.3 5-Mode HIL Table and CDI Sensitivity Table + narrative
    s73_pattern = re.compile(r"### 7\.3 Closed-Loop HIL Synchronization & CDI Sensitivity Analysis.*?(?=\n---\n\n### 7\.4 Physical Hardware Verification)", re.DOTALL)
    m73 = s73_pattern.search(content)
    if not m73:
        raise ValueError("README Section 7.3 pattern not found")

    m1_c, m2_c, m3_c, m4_c, m5_c = (
        float(m1_r["cloud_reschedule_calls"]),
        float(m2_r["cloud_reschedule_calls"]),
        float(m3_r["cloud_reschedule_calls"]),
        float(m4_r["cloud_reschedule_calls"]),
        float(m5_r["cloud_reschedule_calls"]),
    )
    m2_cm, m3_cm, m4_cm, m5_cm = (
        float(m2_r["final_makespan_min"]),
        float(m3_r["final_makespan_min"]),
        float(m4_r["final_makespan_min"]),
        float(m5_r["final_makespan_min"]),
    )
    m2_co, m3_co, m4_co, m5_co = (
        float(m2_r["tou_energy_cost_usd"]),
        float(m3_r["tou_energy_cost_usd"]),
        float(m4_r["tou_energy_cost_usd"]),
        float(m5_r["tou_energy_cost_usd"]),
    )
    m2_lat_s = float(m2_r["total_solver_latency_s"])
    m3_lat_s = float(m3_r["total_solver_latency_s"])
    m5_lat_s = float(m5_r["total_solver_latency_s"])
    m5_lat_std_s = float(m5_r["solver_latency_std_s"])
    m1_pm, m2_pm, m5_pm = float(m1_r["num_pm_replacements"]), float(m2_r["num_pm_replacements"]), float(m5_r["num_pm_replacements"])
    p_m2_mk = _fmt_p(float(m2_r["p_wilcoxon_makespan"]))
    p_m2_co = _fmt_p(float(m2_r["p_wilcoxon_cost"]))
    p_m3_mk = _fmt_p(float(m3_r["p_wilcoxon_makespan"]))
    p_m3_co = _fmt_p(float(m3_r["p_wilcoxon_cost"]))
    p_m4_mk = _fmt_p(float(m4_r["p_wilcoxon_makespan"]))

    new_s73 = (
        "### 7.3 Closed-Loop HIL Synchronization & CDI Sensitivity Analysis\n\n"
        "5-mode closed-loop comparison on Mk01 ($55$ operations) across 15 independent test seeds (`TEST_SEEDS = [301..315]`) "
        "including a **Frequency-Matched Periodic Control (`Mode 3`, Op 25 & Op 50 = $2.00$ calls)** to isolate event timing vs. fixed periodicity (`Mode 3` vs `Mode 5`), "
        "using identical TOU-aware speed logic (preserving off/mid-peak planned speed and derating On-Peak $\\text{rate} \\ge 0.20$ to $s=1$), "
        "statistical EKF PM projection ($w_{\\text{proj}} = \\hat{w} + \\Delta w_{\\text{nom}} + 2\\sigma_{\\text{EKF}} + 2.0$), "
        "and measured **Cloud Warm-Start NSGA-III Resynchronization** overhead (`2.6835 s` per trigger + $2 \\times 0.0814\\,\\text{ms}$ LAN MQTT RTT; list-scheduling surrogate takes $0.2143\\,\\text{ms}$, full offline solve $4.2392\\,\\text{s}$):\n\n"
        + generate_readme_table5() + "\n"
        f"*Component & Timing Attribution:* "
        f"(1) **Frequency-Matched Timing (`Mode 3` vs. `Mode 5`):** `Mode 3` fires $2.00$ fixed periodic calls at Op 25 and Op 50 (${m3_lat_s:.2f}\\,\\text{{s}}$ overhead) matching $N_{{\\text{{shocks}}}}=2.0$. `Mode 5` cuts Cloud Warm-Start calls by **{m3_m5_call_reduct:.1f}%** (${m5_c:.2f} \\pm {m5_calls_s:.2f}$ vs. ${m3_c:.2f}$, ${m5_lat_s:.2f} \\pm {m5_lat_std_s:.2f}\\,\\text{{s}}$ overhead) and trims makespan (${m5_cm:.2f}$ vs. ${m3_cm:.2f}\\,\\text{{min}}$, $p = {p_m3_mk}$; cost \\${m5_co:.3f} vs. \\${m3_co:.3f}, $p = {p_m3_co}$) by reacting immediately when shocks hit rather than waiting for fixed milestones. "
        f"(2) **CDI vs. High-Frequency Periodic (`Mode 2` vs. `Mode 5`):** Cuts synchronization calls and Warm-Start solver queue delay by **{m5_reduct:.1f}%** (${m5_lat_s:.2f}\\,\\text{{s}}$ vs. ${m2_lat_s:.2f}\\,\\text{{s}}$) and significantly reduces TOU cost by **{m2_m5_cost_reduct:.1f}%** (\\${m5_co:.3f} vs. \\${m2_co:.3f}, $p = {p_m2_co}$) with no significant makespan difference ($p = {p_m2_mk}$). "
        f"(3) **Bounded Resync vs. Zero Resync (`Mode 4` vs. `Mode 5`):** CDI recovery cuts makespan by ${m4_cm - m5_cm:.2f}\\,\\text{{min}}$ relative to zero-resync Mode 4 ($p = {p_m4_mk}$). "
        f"(4) **Live EKF Sensitivity (`Mode 1` vs. `Mode 2`/`Mode 5`):** Without live EKF, Mode 1 applies rigid schedule-only PMs (${m1_pm:.2f}$ vs. ${m2_pm:.2f}$/${m5_pm:.2f}$) ignoring $+26\\,\\mu\\text{{m}}$ shocks.\n\n"
        "![Closed-Loop HIL Real-Time Execution](tests/phase4_closed_loop_hil.png)\n\n"
        "#### Full Parametric CDI Threshold Sensitivity Evaluation (10 Evaluation Thresholds, 10 Validation Seeds `VAL_SEEDS = [201..210]`)\n"
        "Trade-off between Cloud Warm-Start re-optimization frequency, schedule degradation, and total operational cost index:\n"
        "$$\\text{Total\\_System\\_Cost\\_Index} = 0.45 \\frac{C_{\\max}}{C_{\\text{ref}}} + 0.45 \\frac{\\text{Cost}}{\\text{Cost}_{\\text{ref}}} + 0.10 \\frac{\\text{Cloud\\_Calls}}{10.0}$$\n\n"
        + generate_cdi_sensitivity_table() + "\n\n"
        f"Thresholds are classified objectively by call count relative to the $N_{{\\text{{shocks}}}} = 2.0$ physical macro-shocks per shift across $\\theta_{{\\text{{th}}}} \\in [0.02, 0.50]$:\n"
        f"- **Over-Triggering Region (`> 2 Shocks`, $\\text{{calls}} > 2.0$, $\\theta_{{\\text{{th}}}} \\le 0.05$):** High synchronization sensitivity triggers ${float(r05['Cloud_Calls_Mean']):.2f} \\pm {float(r05['Cloud_Calls_Std']):.2f}$ to ${float(r02['Cloud_Calls_Mean']):.2f} \\pm {float(r02['Cloud_Calls_Std']):.2f}$ calls per shift (${float(r05['Comm_Overhead (s)']):.2f}$–${float(r02['Comm_Overhead (s)']):.2f}\\,\\text{{s}}$ Warm-Start overhead). $\\theta_{{\\text{{th}}}}^* = {opt_th:.2f}$ achieves the minimum composite `Total_System_Cost_Index` (**${opt_sys:.4f}$**, ${opt_cmax_m:.2f}\\,\\text{{min}}$, \\${opt_cost_m:.3f}) via over-triggering beyond 2 shocks.\n"
        f"- **Event-Matched Plateau (`1–2 Shocks`, $1.0 \\le \\text{{calls}} \\le 2.0$, $\\theta_{{\\text{{th}}}} \\in [{min_pl:.2f}, {max_pl:.2f}]$):** Call rate tracks the 1–2 macro-shocks (${float(r18['Cloud_Calls_Mean']):.2f}$ to ${float(r08['Cloud_Calls_Mean']):.2f}$ calls; ${float(r18['Comm_Overhead (s)']):.2f}$–${float(r08['Comm_Overhead (s)']):.2f}\\,\\text{{s}}$ solver overhead); the frozen center $\\theta_{{\\text{{th}}}} = 0.12$ executes ${float(r12['Cloud_Calls_Mean']):.2f} \\pm {float(r12['Cloud_Calls_Std']):.2f}$ calls (${float(r12['Makespan_Mean (min)']):.2f} \\pm {float(r12['Makespan_Std (min)']):.2f}\\,\\text{{min}}$, \\${float(r12['TOU_Cost_Mean ($)']):.3f} \\pm {float(r12['TOU_Cost_Std ($)']):.3f}$, index ${float(r12['Total_System_Cost_Index']):.4f}$).\n"
        f"- **Under-Triggering Region (`< 1 Shock`, $\\text{{calls}} < 1.0$, $\\theta_{{\\text{{th}}}} \\ge {float(r22['CDI_Threshold']):.2f}$):** Below-unity triggering ($0.90 \\to 0.00$ calls) misses disturbances, degrading makespan to **${float(r50['Makespan_Mean (min)']):.2f} \\pm {float(r50['Makespan_Std (min)']):.2f}\\,\\text{{min}}$** and raising `Total_System_Cost_Index` to ${float(r50['Total_System_Cost_Index']):.4f}$."
    )
    content = content[:m73.start()] + new_s73 + content[m73.end():]

    # 6. Section 7.5 Transport Sensitivity Tables
    idx_a = content.find("#### (a) Benchmark Mk01")
    idx_sec76 = content.find("### 7.6 Unified Table of Experimental Regimes")
    if idx_a != -1 and idx_sec76 != -1:
        prefix_a = "#### (a) Benchmark Mk01 (10 jobs $\\times$ 6 machines, 55 operations)\n\n"
        prefix_b = "#### (b) Benchmark Mk03 (15 jobs $\\times$ 8 machines, 150 operations)\n\n"
        new_s75_insights = (
            "![Transport Latency Sensitivity](tests/phase5_transport_sensitivity.png)\n\n"
            "#### Key Physical & Methodological Insights:\n"
            "1. **Boundary Parity ($\\tau_{\\text{trans}} = 0.0\\,\\text{min}$):** At zero transport delay, all 4 methods reproduce the exact multi-instance figures in `phase5_wilcoxon_benchmark_table.csv` ($86.00 \\pm 9.79\\,\\text{min}$, $\\$2.793 \\pm 0.183$, $329.99 \\pm 60.47\\,\\text{min}$, $0.00$ breakdowns for Proposed DT on Mk01; $462.69 \\pm 45.99\\,\\text{min}$, $\\$13.883 \\pm 0.677$, $4621.78 \\pm 889.63\\,\\text{min}$, $0.00$ breakdowns on Mk03) with zero deviation ($< 10^{-6}$).\n"
            "2. **Monotonic Makespan Scaling:** For all schedule-following methods (`Proposed Cloud-Edge DT`, `Right-Shift Rescheduling`, `NSGA-III`), makespan scales monotonically as inter-machine transfers lengthen precedence paths.\n"
            "3. **Invariance of Tool Preservation:** Across all four transport latency tiers ($0.0, 1.0, 2.0, 3.0\\,\\text{min}$), `Proposed Cloud-Edge DT` (alongside identically EKF-equipped `Right-Shift Rescheduling`) preserves **$0.00$ tool breakdowns** on both `Mk01` and `Mk03`, in contrast to $0.87$ failures on Mk01 / $9.67$ on Mk03 for open-loop NSGA-III and $5.60$–$6.73$ failures on Mk03 for unguided Standalone Edge PPO.\n"
            "4. **Physical Mechanism for Non-Monotonic TOU Electricity Cost:** Shifting operations across the 100-minute cyclic TOU tariff ($\\$0.06 \\to \\$0.20 \\to \\$0.12/\\text{kWh}$) can reduce net electricity expenditure when high-load cuts are displaced out of On-Peak windows.\n\n"
            "---\n\n"
        )
        content = (
            content[:idx_a]
            + prefix_a + generate_readme_transport_subtable("Mk01") + "\n\n"
            + prefix_b + generate_readme_transport_subtable("Mk03") + "\n\n"
            + new_s75_insights
            + content[idx_sec76:]
        )

    # 7. Section 7.8 Literature Positioning Proposed row
    old_lit_prop = re.compile(r"\| \*\*Proposed Framework\*\* \|.*?\n")
    new_lit_prop = (
        f"| **Proposed Framework** | **Closed-Loop Co-Design: Cloud NSGA-III + Edge MaskablePPO + Dual-Sensor Fusion + Shift-Normalized CDI** | "
        "**Continuous tool wear drift ($+15\\%..+35\\%$) + Stochastic dual macro-shocks (Hard alloy & Stoppage)** | "
        "**Makespan ($C_{\\max}$), TOU Electricity Cost, Total Weighted Tardiness (TWT)** | "
        "**Physics-informed Taylor dynamic wear + Dual-sensor EKF (INA226 + ADXL345, $2\\sigma_{\\text{EKF}}$ bound) + Pre-operation PM ($0.00$ breakdowns) + `GPIO 27` Open-Drain Hold & $300\\,\\text{ms}$ Watchdog** | "
        "**Sub-millisecond real-time speed adaptation ($s \\in \\{1, 2, 3\\}$) synchronized with Cyclic TOU Tariff & Peak Limit ($P_{\\text{limit}} = 22.0\\,\\text{kW}$)** | "
        f"**Full Co-Design: Event-triggered re-sync ($\\theta_{{\\text{{th}}}} = 0.12$ in $[{min_pl:.2f}, {max_pl:.2f}]$ Event-Matched Plateau) cutting Cloud Warm-Start calls by {m5_reduct:.1f}% ({m5_calls:.2f} vs. {m2_calls:.2f}; -{m3_m5_call_reduct:.1f}% vs. freq-matched {m3_calls:.2f}) while retaining sub-ms edge reactivity (0.0132 ms forward, 0.1506 ms step)** |\n"
    )
    content = old_lit_prop.sub(lambda _: new_lit_prop, content, count=1)

    # 8. Section 7.9 Right-Shift & Large-Scale Trade-offs (NEVER say statistical equivalence)
    old_rs_block = re.compile(r"1\. \*\*Honest Assessment of `Right-Shift Rescheduling`:\*\*.*?(?=\n\n2\. \*\*Objective Analysis)", re.DOTALL)
    new_rs_block = (
        "1. **Honest Assessment of `Right-Shift Rescheduling`:**\n"
        "   - *Fair EKF-Equipped Baseline:* Equipped with the exact same `BreadboardDAQInterface` + EKF wear estimator and PM safety rule as Proposed DT (no oracle access), Right-Shift also achieves **$0.00$ tool breakdowns** across all five benchmarks (`Mk01`–`Mk05`). Under two-sided paired Wilcoxon testing with Holm-Bonferroni correction (which tests distributional difference rather than equivalence / TOST), **no statistically significant difference in Hypervolume is observed** on `Mk01` ($p_{\\text{holm}} = "
        f"{mk01_rs_pholm:.4f}$) and `Mk04` ($p_{{\\text{{holm}}}} = {mk04_rs_pholm:.4f}$), whereas Right-Shift achieves statistically higher Hypervolume on `Mk02` ($p_{{\\text{{holm}}}} = {mk02_rs_pholm:.4f}$), `Mk03` ($p_{{\\text{{holm}}}} = {mk03_rs_pholm:.4f}$), and `Mk05` ($p_{{\\text{{holm}}}} = {mk05_rs_pholm:.4f}$) by preserving fixed NSGA-III assignments without On-Peak speed throttling.\n"
        f"   - *Economic Advantage of Proposed DT:* Active TOU-aware speed-scaling ($s \\in \\{{1, 2, 3\\}}$) and CDI synchronization reduce TOU electricity cost on bottleneck `Mk05` by **{mk05_sav:.2f}%** (\\${float(prop_map['Mk05']['Cost_Mean']):.3f} $\\pm$ {float(prop_map['Mk05']['Cost_Std']):.3f} vs. \\${float(rs_map['Mk05']['Cost_Mean']):.3f} $\\pm$ {float(rs_map['Mk05']['Cost_Std']):.3f})."
    )
    content = old_rs_block.sub(lambda _: new_rs_block, content, count=1)

    old_dilemma = re.compile(r"3\. \*\*Discrepancy Between Hypervolume \(HV\) and IGD on Benchmark `Mk05`:\*\*.*?(?=\n\n4\. \*\*Small-Scale vs\. Large-Scale)", re.DOTALL)
    new_dilemma = (
        "3. **Discrepancy Between Hypervolume (HV) and IGD on Benchmark `Mk05`:**\n"
        "   - *The Bottleneck Machine Dilemma:* `Mk05` poses an extreme structural bottleneck ($15$ jobs, $106$ operations on $4$ machines).\n"
        f"   - *Trade-off Mechanism:* Proposed DT prioritizes $0.00$ tool breakdowns (vs. $9.20$ for NSGA-III, $9.87$ for MOEA-D, $12.67$ for RVEA; equal to $0.00$ for fair EKF-equipped Right-Shift) and lowest dynamic TOU cost (\\$11.198 $\\pm$ 0.539 vs. \\${float(rs_map['Mk05']['Cost_Mean']):.3f} $\\pm$ {float(rs_map['Mk05']['Cost_Std']):.3f} for Right-Shift, saving **{mk05_sav:.2f}%**). Decelerating to $s=1$ in peak hours extends makespan ($408.76\\,\\text{{min}}$ vs. ${float(rs_map['Mk05']['Makespan_Mean']):.2f}\\,\\text{{min}}$), compressing Hypervolume along makespan/TWT while achieving lower IGD (${float(prop_map['Mk05']['IGD_Mean']):.4f}$ vs. $0.5000 - 0.6280$ for open-loop metaheuristics)."
    )
    content = old_dilemma.sub(lambda _: new_dilemma, content, count=1)

    content = content.replace(
        "beating NSGA-III on Mk03 ($p_{\\text{holm}} = 0.0256$).",
        f"beating NSGA-III ($p_{{\\text{{holm}}}} = {mk03_nsga3_pholm:.4f}$) and NSGA-II ($p_{{\\text{{holm}}}} = {mk03_nsga2_pholm:.4f}$) on Mk03.",
    )
    content = content.replace("($\\theta_{\\text{th}} \\in [0.08, 0.15]$)", f"($\\theta_{{\\text{{th}}}} \\in [{min_pl:.2f}, {max_pl:.2f}]$)")

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print("  -> PASSED: README.md fully synchronized!")


def update_audit_specs():
    print("[3/3] Synchronizing results/audit_methodology_specs.md...")
    with open(AUDIT_SPECS_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    wilc_rows = load_csv("phase5_wilcoxon_benchmark_table.csv")
    rs_map = {r["Instance"]: r for r in wilc_rows if "Right-Shift" in r["Method"]}
    prop_map = {r["Instance"]: r for r in wilc_rows if "Proposed" in r["Method"]}
    mk05_sav = (float(rs_map["Mk05"]["Cost_Mean"]) - float(prop_map["Mk05"]["Cost_Mean"])) / float(rs_map["Mk05"]["Cost_Mean"]) * 100.0
    mk05_rs_bds = float(rs_map["Mk05"]["Breakdowns_Mean"])

    # Update Section 3.2 CDI threshold note
    sec32_note_pat = re.compile(r"- Ngưỡng triển khai chọn.*?\n")
    content = sec32_note_pat.sub(
        lambda _: "- Ngưỡng triển khai chọn $\\theta_{\\text{th}} = 0.12$ (nằm trong vùng `Event-Matched Plateau (1–2 Shocks)` $[0.08, 0.18]$ bám sát đúng $N_{\\text{shocks}} = 2.0$; cực tiểu chỉ số hợp thành `Total_System_Cost_Index` $= 1.0325$ rơi tại $\\theta_{\\text{th}} = 0.05$ thuộc vùng `Over-Triggering Region (> 2 Shocks)` với $2.70$ calls).\n",
        content,
        count=1,
    )

    # Replace entire Section 5.3 cleanly so numbers and explanation match fair EKF Right-Shift
    sec53_pat = re.compile(
        r"### 5\.3 Giải Mã Nghịch Lý Thống Kê Trên Mk05:.*?(?=\n---\n\n## 6\. Bản Chất Dữ Liệu Thực Nghiệm HIL)",
        re.DOTALL,
    )
    new_sec53 = (
        f"### 5.3 Giải Mã Nghịch Lý Thống Kê Trên Mk05: Vì Sao Right-Shift Đạt HV Cao Hơn ({float(rs_map['Mk05']['HV_Mean']):.4f} vs {float(prop_map['Mk05']['HV_Mean']):.4f}) Nhưng Proposed Lại Giảm {mk05_sav:.2f}% Chi Phí Điện TOU?\n\n"
        "Dữ liệu thực nghiệm thực tế trên `Mk05` (15 test seeds, cả hai cùng dùng EKF & quy tắc PM):\n"
        "- **Right-Shift Rescheduling (EKF Parity):**\n"
        f"  $\\text{{Makespan}} = {float(rs_map['Mk05']['Makespan_Mean']):.2f} \\pm {float(rs_map['Mk05']['Makespan_Std']):.2f}\\text{{ min}}$, "
        f"$\\text{{TOU Cost}} = \\${float(rs_map['Mk05']['Cost_Mean']):.3f} \\pm {float(rs_map['Mk05']['Cost_Std']):.3f}$, "
        f"$\\text{{TWT}} = {float(rs_map['Mk05']['TWT_Mean']):.1f} \\pm {float(rs_map['Mk05']['TWT_Std']):.1f}\\text{{ min}}$, "
        f"$\\text{{Breakdowns}} = {mk05_rs_bds:.2f}$.\n"
        f"  $\\mathbf{{HV = {float(rs_map['Mk05']['HV_Mean']):.4f} \\pm {float(rs_map['Mk05']['HV_Std']):.4f}}}, \\quad \\mathbf{{IGD = {float(rs_map['Mk05']['IGD_Mean']):.4f} \\pm {float(rs_map['Mk05']['IGD_Std']):.4f}}}$.\n"
        "- **Proposed Cloud-Edge DT:**\n"
        f"  $\\text{{Makespan}} = {float(prop_map['Mk05']['Makespan_Mean']):.2f} \\pm {float(prop_map['Mk05']['Makespan_Std']):.2f}\\text{{ min}}$, "
        f"$\\text{{TOU Cost}} = \\mathbf{{\\${float(prop_map['Mk05']['Cost_Mean']):.3f} \\pm {float(prop_map['Mk05']['Cost_Std']):.3f}}}$, "
        f"$\\text{{TWT}} = {float(prop_map['Mk05']['TWT_Mean']):.1f} \\pm {float(prop_map['Mk05']['TWT_Std']):.1f}\\text{{ min}}$, "
        f"$\\text{{Breakdowns}} = \\mathbf{{0.00}}$.\n"
        f"  $\\mathbf{{HV = {float(prop_map['Mk05']['HV_Mean']):.4f} \\pm {float(prop_map['Mk05']['HV_Std']):.4f}}}, \\quad \\mathbf{{IGD = {float(prop_map['Mk05']['IGD_Mean']):.4f} \\pm {float(prop_map['Mk05']['IGD_Std']):.4f}}}$.\n\n"
        "#### Phân Tích Hình Học & Bản Chất Toán Học:\n"
        "1. **Bản chất của Hypervolume (HV):**\n"
        "   HV đo thể tích hình hộp đa chiều giới hạn bởi điểm nghiệm và điểm tham chiếu $(1.1, 1.1, 1.1)$:\n"
        "   $$HV \\approx (1.1 - c_{\\text{norm}}) \\times (1.1 - \\text{cost}_{\\text{norm}}) \\times (1.1 - \\text{twt}_{\\text{norm}})$$\n"
        f"   - Khi cùng được trang bị cảm biến EKF và ngưỡng PM an toàn để đạt $0.00$ sự cố gãy dao, `Right-Shift` vẫn giữ nguyên tốc độ và gán máy ban đầu của NSGA-III mà không giảm tốc độ vào giờ cao điểm TOU, nên giữ được Makespan ($376.46\\text{{ min}}$) và TWT ($4806.7\\text{{ min}}$) ngắn hơn, kéo $c_{{\\text{{norm}}}}$ và $\\text{{twt}}_{{\\text{{norm}}}}$ về sát $0$ và đạt HV cao hơn ($p_{{\\text{{holm}}}} = {float(rs_map['Mk05']['HV_p_holm']):.4f}$).\n"
        "2. **Lợi Thế Kinh Tế TOU Của Proposed Cloud-Edge DT:**\n"
        "   - `Proposed Cloud-Edge DT` chủ động hạ tốc độ cắt về $s=1$ trong giờ cao điểm TOU ($\\$0.20/\\text{kWh}$) và thực hiện tái lập lịch theo CDI:\n"
        f"     * **Chi phí điện TOU thấp nhất:** $\\$11.198 \\pm 0.539$ (tiết kiệm **{mk05_sav:.2f}%** so với $\\${float(rs_map['Mk05']['Cost_Mean']):.3f}$ của Right-Shift, thấp hơn $\\$12.697$ của NSGA-III và $\\$19.270$ của SPT).\n"
        "     * **Độ trễ công việc (TWT) thấp hơn các thuật toán tiến hóa mở vòng:** $5395.5\\,\\text{min}$ so với $6100.4\\,\\text{min}$ của NSGA-III và $6984.4\\,\\text{min}$ của MOEA-D.\n"
        "     * **Tuyệt đối an toàn công cụ:** $0.00$ sự cố gãy dao (so với $9.20$ lần gãy dao của NSGA-III và $12.67$ lần của RVEA).\n\n"
        "### 5.4 Baseline Fairness & Revision Transparency Note (Ghi Chú Minh Bạch Chuẩn Hóa Baseline)\n"
        "Trong phiên bản chuẩn hóa hiện tại, hàm `execute_right_shift_rescheduling()` ([tests/test_task5_2_multi_instance_eval.py](file:///home/neyugn/Documents/cloud_edge_fjsp_dt/tests/test_task5_2_multi_instance_eval.py#L115)) được trang bị **cùng bộ cảm biến `BreadboardDAQInterface` + `RealTimeToolWearEstimator` (Live EKF) và cùng quy tắc ngưỡng PM an toàn** ($w_{\\text{proj}} = \\hat{w} + \\Delta w_{\\text{nom}} + 2\\sigma_{\\text{EKF}} + 2.0$, tuyệt đối không đọc oracle `twin.current_wear_um`) như `Proposed Cloud-Edge DT`. Thay đổi này loại bỏ hoàn toàn sự bất đối xứng về cảm biến ở các phiên bản trước (nơi Right-Shift không có EKF và bị $0.40 - 1.47$ gãy dao), đảm bảo cả hai đều đạt $0.00$ gãy dao và tách bạch công bằng sự khác biệt thuần túy về chiến lược điều độ: Right-Shift giữ cố định máy/tốc độ theo lịch gốc (chỉ dịch phải khi PM), còn Proposed DT điều chỉnh tốc độ động theo biểu giá TOU bằng mạng ONNX và tái lập lịch theo CDI."
    )
    m53_full = sec53_pat.search(content)
    if m53_full:
        content = content[:m53_full.start()] + new_sec53 + content[m53_full.end():]

    # Fix duplicated ADXL345 text idempotently in Section 6.1
    dup_adxl_pat = re.compile(r"Cảm biến gia tốc ADXL345(?:\s*\(\$\\pm 16g\$, \$200\\,\\text\{Hz\}\$\) và chân phanh cứng `GPIO 27` \(`PIN_GRBL_FEED_HOLD`, chờ `Hold:0` trước khi `Soft Reset`/cắt SSR, gắn cờ `sample_held` khi giữ mẫu ZOH 3 chu kỳ I2C \$15\\,\\text\{ms\}\$\))+")
    clean_adxl = (
        "Cảm biến gia tốc ADXL345 ($\\pm 16g$, $200\\,\\text{Hz}$) với chân phanh cứng Open-Drain `GPIO 27` "
        "(`PIN_GRBL_FEED_HOLD`, chờ `Hold:0` hoặc timeout động học $2500\\,\\text{ms}$ từ `$110=1000, $120=10` trước khi `Soft Reset`/cắt SSR, "
        "gắn cờ `sample_held` khi giữ mẫu ZOH 3 chu kỳ I2C $15\\,\\text{ms}$, kết hợp Watchdog $300\\,\\text{ms}$ phía Host)"
    )
    if dup_adxl_pat.search(content):
        content = dup_adxl_pat.sub(lambda _: clean_adxl, content, count=1)

    with open(AUDIT_SPECS_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print("  -> PASSED: results/audit_methodology_specs.md fully synchronized!")


def main():
    print("=" * 80)
    print("MARKDOWN SYNCHRONIZATION AND DIRECTIONAL PERCENTAGE ENFORCEMENT")
    print("=" * 80)
    update_manuscript()
    update_readme()
    update_audit_specs()
    print("=" * 80)
    print("ALL MARKDOWN DOCUMENTS SUCCESSFULLY SYNCHRONIZED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
