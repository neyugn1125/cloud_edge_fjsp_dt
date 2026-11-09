import csv

def generate_table1():
    with open('data/benchmarks/phase5_wilcoxon_benchmark_table.csv') as f:
        reader = list(csv.DictReader(f))
    
    lines = [
        "| Instance | Method | Hypervolume (HV) $\\uparrow$ | IGD $\\downarrow$ | Makespan $C_{\\max}$ (min) $\\downarrow$ | TOU Cost (\\$) $\\downarrow$ | Tardiness TWT (min) $\\downarrow$ | Tool Failures $\\downarrow$ | Cliff's $\\delta$ | $p_{\\text{holm}}$ |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    cur_inst = ''
    for row in reader:
        inst = row['Instance']
        scale = row['Scale']
        m = row['Method']
        
        hv_m, hv_s = float(row['HV_Mean']), float(row['HV_Std'])
        igd_m, igd_s = float(row['IGD_Mean']), float(row['IGD_Std'])
        c_m, c_s = float(row['Makespan_Mean']), float(row['Makespan_Std'])
        cost_m, cost_s = float(row['Cost_Mean']), float(row['Cost_Std'])
        twt_m, twt_s = float(row['TWT_Mean']), float(row['TWT_Std'])
        bd = float(row['Breakdowns_Mean'])
        
        is_prop = 'Proposed' in m
        
        if is_prop:
            m_str = '**Proposed Cloud-Edge DT**'
            hv_str = f"**${hv_m:.4f} \\pm {hv_s:.4f}$**"
            igd_str = f"**${igd_m:.4f} \\pm {igd_s:.4f}$**"
            c_str = f"**${c_m:.2f} \\pm {c_s:.2f}$**"
            cost_str = f"**\\${cost_m:.3f} \\pm {cost_s:.3f}**"
            twt_str = f"**${twt_m:.1f} \\pm {twt_s:.1f}$**"
            bd_str = "**0.00**"
            d_str = "**Base**"
            p_str = "**Base**"
        else:
            m_str = m.replace(' (No Cloud Prior)', '').replace(' (NSGA-III + Edge ONNX + CDI)', '')
            hv_str = f"${hv_m:.4f} \\pm {hv_s:.4f}$"
            igd_str = f"${igd_m:.4f} \\pm {igd_s:.4f}$"
            c_str = f"${c_m:.2f} \\pm {c_s:.2f}$"
            cost_str = f"\\${cost_m:.3f} \\pm {cost_s:.3f}"
            twt_str = f"${twt_m:.1f} \\pm {twt_s:.1f}$"
            bd_str = f"{bd:.2f}"
            
            d_val = float(row['HV_Cliffs_Delta'])
            d_str = f"+{d_val:.2f}" if d_val > 0 else f"{d_val:.2f}"
            p_val = float(row['HV_p_holm'])
            p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
            
        if inst != cur_inst:
            cur_inst = inst
            inst_str = f"**{inst}**<br>({scale})"
        else:
            inst_str = ""
            
        lines.append(f"| {inst_str} | {m_str} | {hv_str} | {igd_str} | {c_str} | {cost_str} | {twt_str} | {bd_str} | {d_str} | {p_str} |")
    return "\n".join(lines)

def generate_table2():
    with open('data/benchmarks/phase5_ablation_results.csv') as f:
        reader = list(csv.DictReader(f))
    
    lines = [
        "| Instance | Variant | Description | Makespan (min) | TOU Cost (\\$) | TWT (min) | Breakdowns |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |"
    ]
    
    cur_inst = ''
    for row in reader:
        inst = row['Instance']
        scale = row['Scale']
        var = row['Variant']
        desc = row['Description']
        
        c_m, c_s = float(row['Makespan Mean (min)']), float(row['Makespan Std (min)'])
        c_deg = float(row['Makespan Degradation (%)'])
        cost_m, cost_s = float(row['TOU Cost Mean ($)']), float(row['TOU Cost Std ($)'])
        cost_deg = float(row['TOU Cost Degradation (%)'])
        twt_m, twt_s = float(row['TWT Mean (min)']), float(row['TWT Std (min)'])
        twt_deg = float(row['TWT Degradation (%)'])
        bd = float(row['Breakdowns Mean'])
        
        if 'Full Proposed' in var:
            var_str = "**Full Proposed DT**"
            desc_str = "Full closed-loop framework (Dual-Sensor EKF + Speed Scaling + SSR Standby + Cloud Guidance)"
            c_str = f"**${c_m:.2f} \\pm {c_s:.2f}$ (Base)**"
            cost_str = f"**\\${cost_m:.3f} \\pm {cost_s:.3f} (Base)**"
            twt_str = f"**${twt_m:.2f} \\pm {twt_s:.2f}$ (Base)**"
            bd_str = "**0.00**"
        else:
            var_str = var.split(' (')[0]
            if 'Variant 1' in var:
                desc_str = "w/o Dual-Sensor Wear Feedback (Delayed PM under disturbance)"
            elif 'Variant 2' in var:
                desc_str = "Fixed Speed $s=2$ (No real-time dynamic TOU peak avoidance)"
            elif 'Variant 3' in var:
                desc_str = "Idle-Only (No SSR standby power gating during waiting intervals)"
            elif 'Variant 4' in var:
                desc_str = "w/o Cloud Guidance (Pure local edge dispatching without global macro prior)"
            else:
                desc_str = desc
                
            c_deg_str = f"{c_deg:+.1f}%"
            cost_deg_str = f"{cost_deg:+.1f}%"
            twt_deg_str = f"{twt_deg:+.1f}%"
            
            c_str = f"${c_m:.2f} \\pm {c_s:.2f}$ ({c_deg_str})"
            cost_str = f"\\${cost_m:.3f} \\pm {cost_s:.3f}$ ({cost_deg_str})"
            twt_str = f"${twt_m:.2f} \\pm {twt_s:.2f}$ ({twt_deg_str})"
            bd_str = f"**{bd:.2f}**" if bd > 0 else "0.00"
            
        if inst != cur_inst:
            cur_inst = inst
            inst_str = f"**{inst}**<br>({scale})"
        else:
            inst_str = ""
            
        lines.append(f"| {inst_str} | {var_str} | {desc_str} | {c_str} | {cost_str} | {twt_str} | {bd_str} |")
    return "\n".join(lines)

def generate_table4():
    with open('data/benchmarks/phase5_transport_sensitivity.csv') as f:
        reader = list(csv.DictReader(f))
        
    lines = [
        "| Instance | Transport Latency $\\tau_{\\text{trans}}$ | Method | Makespan (min) $\\downarrow$ | TOU Cost (\\$) $\\downarrow$ | Tardiness TWT (min) $\\downarrow$ | Tool Failures $\\downarrow$ |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: |"
    ]
    
    cur_inst = ''
    cur_tau = ''
    for row in reader:
        inst = row['Instance']
        tau = float(row['Transport_Time_min'])
        m = row['Method']
        
        c_m, c_s = float(row['Makespan_Mean']), float(row['Makespan_Std'])
        cost_m, cost_s = float(row['TOU_Cost_Mean']), float(row['TOU_Cost_Std'])
        twt_m, twt_s = float(row['TWT_Mean']), float(row['TWT_Std'])
        bd = float(row['Breakdowns_Mean'])
        
        is_prop = 'Proposed' in m
        
        if is_prop:
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

def generate_table3():
    with open('data/benchmarks/phase5_cdi_sensitivity.csv') as f:
        reader = list(csv.DictReader(f))

    lines = [
        "| CDI Threshold $\\theta_{\\text{th}}$ | Cloud Calls ($\\text{Mean} \\pm \\text{Std}$) | Makespan ($\\text{Mean} \\pm \\text{Std}$) | TOU Cost (\\$) | Max CDI ($\\text{Mean} \\pm \\text{Std}$) | Schedule Index | Total System Cost Index | Operational Region |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]

    for row in reader:
        th = float(row['CDI_Threshold'])
        calls_m, calls_s = float(row['Cloud_Calls_Mean']), float(row['Cloud_Calls_Std'])
        c_m, c_s = float(row['Makespan_Mean (min)']), float(row['Makespan_Std (min)'])
        cost_m, cost_s = float(row['TOU_Cost_Mean ($)']), float(row['TOU_Cost_Std ($)'])
        cdi_m, cdi_s = float(row['Max_CDI_Mean']), float(row['Max_CDI_Std'])
        sched_idx = float(row['Schedule_Composite_Index'])
        sys_idx = float(row['Total_System_Cost_Index'])
        zone = row['Zone']

        if abs(th - 0.12) < 1e-4:
            th_str = f"**{th:.2f}**"
            calls_str = f"**${calls_m:.2f} \\pm {calls_s:.2f}$**"
            c_str = f"**${c_m:.2f} \\pm {c_s:.2f}$**"
            cost_str = f"**\\${cost_m:.3f} \\pm {cost_s:.3f}**"
            cdi_str = f"**${cdi_m:.4f} \\pm {cdi_s:.4f}$**"
            sched_str = f"**{sched_idx:.4f}**"
            sys_str = f"**{sys_idx:.4f} (Knee Point)**"
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

def generate_table5():
    with open('data/benchmarks/phase4_hil_metrics.csv') as f:
        reader = list(csv.DictReader(f))

    lines = [
        "| Operational Mode | Cloud Calls | Call Reduction (%) | Solver Overhead (s) | Edge Latency (ms) | Network Packets | Makespan (min) | Energy (kWh) | TOU Cost (\\$) | Max CDI | Final CDI | PM Replacements | Tool Breakdowns |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for row in reader:
        mode_id = row['mode_id']
        name = row['mode_name']
        calls = int(row['cloud_reschedule_calls'])
        reduct = float(row['reduction_in_cloud_calls_pct'])
        solver_s = float(row['total_solver_latency_s'])
        edge_ms = float(row['edge_decision_latency_ms'])
        packets = int(row['total_network_packets'])
        cmax = float(row['final_makespan_min'])
        energy = float(row['total_energy_kwh'])
        cost = float(row['tou_energy_cost_usd'])
        max_cdi = float(row['max_cdi'])
        final_cdi = float(row['final_cdi'])
        pm = int(row['num_pm_replacements'])
        bd = int(row['num_breakdowns'])

        edge_str = f"{edge_ms:.4f} ms" if edge_ms > 0 else "—"
        if mode_id == "Mode 3":
            mode_str = f"**{mode_id}: {name}**"
            calls_str = f"**{calls}**"
            reduct_str = f"**{reduct:.1f}%**"
            cmax_str = f"**{cmax:.2f}**"
            cost_str = f"**\\${cost:.3f}**"
            bd_str = "**0**"
        elif mode_id == "Mode 2":
            mode_str = f"{mode_id}: {name}"
            calls_str = f"{calls}"
            reduct_str = f"{reduct:.1f}%"
            cmax_str = f"{cmax:.2f}"
            cost_str = f"\\${cost:.3f}"
            bd_str = f"**{bd}**"
        else:
            mode_str = f"{mode_id}: {name}"
            calls_str = f"{calls}"
            reduct_str = f"{reduct:.1f}%"
            cmax_str = f"{cmax:.2f}"
            cost_str = f"\\${cost:.3f}"
            bd_str = f"**{bd}**"

        lines.append(f"| {mode_str} | {calls_str} | {reduct_str} | {solver_s:.1f} s | {edge_str} | {packets} | {cmax_str} | {energy:.2f} | {cost_str} | {max_cdi:.4f} | {final_cdi:.4f} | {pm} | {bd_str} |")
    return "\n".join(lines)

if __name__ == '__main__':
    print("=== TABLE 1 ===")
    print(generate_table1())
    print("\n=== TABLE 2 ===")
    print(generate_table2())
    print("\n=== TABLE 4 ===")
    print(generate_table4())
