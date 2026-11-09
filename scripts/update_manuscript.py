import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.format_manuscript_tables import (
    generate_table1,
    generate_table2,
    generate_table3,
    generate_table4,
    generate_table5,
)

def main():
    with open('results/manuscript_draft.md') as f:
        content = f.read()

    # 1. Update Abstract: standby energy percentage
    content = content.replace("cuts standby energy costs by up to $+38.21\%$", "cuts standby energy costs by up to $+44.82\%$")

    # 2. Update Table 1
    t1_pattern = re.compile(r"(\| Instance \| Method \| Hypervolume \(HV\).*?\n)(.*?)(?=\n### 3\.3 Critical Examination)", re.DOTALL)
    m = t1_pattern.search(content)
    if not m:
        raise ValueError("Table 1 pattern not found")
    new_t1 = generate_table1()
    content = content[:m.start()] + new_t1 + content[m.end():]

    # 3. Update Table 2
    t2_pattern = re.compile(r"(\| Instance \| Variant \| Description \| Makespan.*?\n)(.*?)(?=\n### 4\.2 Analytical Findings)", re.DOTALL)
    m2 = t2_pattern.search(content)
    if not m2:
        raise ValueError("Table 2 pattern not found")
    new_t2 = generate_table2()
    content = content[:m2.start()] + new_t2 + content[m2.end():]

    # 4. Update Table 3 and CDI narrative
    t3_pattern = re.compile(r"(\| CDI Threshold \$\\theta_\{.*?)(?=\nAs shown in Figure 4\(b\))", re.DOTALL)
    m3 = t3_pattern.search(content)
    if m3:
        new_t3 = generate_table3()
        content = content[:m3.start()] + new_t3 + content[m3.end():]

    old_cdi_bullets = (
        "- **Over-Triggering Region ($\\theta_{\\text{th}} \\le 0.05$):** The system triggers up to $46.90 \\pm 1.92$ Cloud rescheduling calls, incurring significant communication/solver latency ($125.9\\,\\text{s}$) and inflating the total cost index by $+52.5\\%$ ($1.6004$).\n"
        "- **Robust Operating Plateau ($\\theta_{\\text{th}} \\in [0.08, 0.15]$):** Cloud calls stabilize at exactly $2.00 \\pm 0.00$ (responding selectively to the two discrete macro-disturbances). Makespan stays at optimal $64.44 \\pm 12.80\\,\\text{min}$, and the total cost index reaches its global minimum ($1.0491$) at $\\theta_{\\text{th}} = 0.12$.\n"
        "- **Uncorrected Drift Region ($\\theta_{\\text{th}} \\ge 0.18$):** Macro-shocks fail to trigger Cloud re-optimization, causing makespan to degrade by $+39.9\\%$ ($90.15\\,\\text{min}$) and TOU costs to rise by $+13.4\\%$ ($\\$3.256$)."
    )
    new_cdi_bullets = (
        "- **Over-Triggering Region ($\\theta_{\\text{th}} \\le 0.05$):** The system triggers up to $44.30 \\pm 2.10$ Cloud rescheduling calls, incurring high solver wait latency ($118.9\\,\\text{s}$) and inflating the total system cost index by $+40.8\\%$ ($1.4931$).\n"
        "- **Robust Operating Plateau ($\\theta_{\\text{th}} \\in [0.08, 0.15]$):** Cloud calls stabilize at $2.00 \\pm 0.00$ to $2.20 \\pm 0.40$ (responding selectively to the two genuine macro-disturbances). Makespan remains optimal at $68.09 \\pm 7.31\\,\\text{min}$, and the total system cost index achieves its global minimum ($1.0607$) at $\\theta_{\\text{th}} = 0.12$.\n"
        "- **Uncorrected Drift Region ($\\theta_{\\text{th}} \\ge 0.18$):** Macro-shocks fail to trigger timely Cloud re-optimization, causing makespan to degrade up to $80.15 \\pm 4.31\\,\\text{min}$ ($+17.7\\%$) and TOU costs to rise to $\\$3.100 \\pm 0.067$ ($+11.5\\%$)."
    )
    content = content.replace(old_cdi_bullets, new_cdi_bullets)

    # 5. Update Table 4
    t4_pattern = re.compile(r"(\| Instance \| Transport Latency.*?\n)(.*?)(?=\n#### Key Findings from Transport Sensitivity:)", re.DOTALL)
    m4 = t4_pattern.search(content)
    if not m4:
        raise ValueError("Table 4 pattern not found")
    new_t4 = generate_table4()
    content = content[:m4.start()] + new_t4 + content[m4.end():]

    # 6. Update or Add Section 4.5 Closed-Loop HIL
    sec45 = (
        "### 4.5 Closed-Loop Multi-Tier Latency & Three-Mode Operational Regimes Evaluation\n"
        "To evaluate the runtime viability and synchronization overhead of the closed-loop collaborative architecture under stochastic physical disruptions, a full Hardware-in-the-Loop (HIL) simulation was conducted across 15 independent test seeds (Brandimarte `Mk01` physical twin, 55 operations) comparing three operational modes:\n"
        "- **Mode 1 (Periodic Cloud Rescheduling):** Rescheduling triggered periodically every 5 operations without edge intelligence.\n"
        "- **Mode 2 (One-Shot Cloud + Edge Autonomous):** Initial Cloud schedule executed with standalone edge dispatching without Cloud re-sync (0 Cloud calls).\n"
        "- **Mode 3 (Event-Triggered Cloud-Edge DT):** Proposed collaborative architecture triggering Cloud re-sync dynamically when $\\text{CDI}(t) > \\theta_{\\text{th}} = 0.12$.\n\n"
        "```\n"
        "Table 5: Quantitative Comparison of Three Operational Regimes in Closed-Loop HIL Simulation (55 Operations, Mk01 Physical Twin)\n"
        "```\n"
        + generate_table5() + "\n\n"
        "#### Multi-Tier Latency Hierarchy & Multi-Dimensional Speedup:\n"
        "Empirical benchmarking on a Linux x86_64 workstation (Python 3.12, ONNX Runtime CPUExecutionProvider) confirms the multi-tier timing distribution:\n"
        "1. **Edge ONNX Forward Pass:** $0.0132\\,\\text{ms}$ ($13.2\\,\\mu\\text{s}$, $\\sim 75,500$ inferences/second).\n"
        "2. **Edge Decision Step:** $0.1506\\,\\text{ms}$ (full feature extraction, action masking, and ONNX prediction).\n"
        "3. **Local MQTT Transport RTT:** $0.0814\\,\\text{ms}$ (industrial LAN broker loopback).\n"
        "4. **Edge Shift Computation:** $7.955\\,\\text{ms}$ (entire 55-step shift simulation on Mk01).\n"
        "5. **Cloud NSGA-III Resync Solve:** $2.6835\\,\\text{s}$ (sub-problem re-optimization).\n"
        "6. **Cloud NSGA-III Full Solve:** $4.2392\\,\\text{s}$ ($4.24 \\pm 0.03\\,\\text{s}$, global schedule generation on Mk01).\n\n"
        "This establishes a clear multi-dimensional speedup advantage:\n"
        "- **Per-Decision Real-Time Reactivity:** The edge neural controller makes complete dispatching decisions in $0.1506\\,\\text{ms}$, which is $\\sim 28,147\\times$ ($\\sim 2.81 \\times 10^4\\times$) faster than a full Cloud solve ($4.2392\\,\\text{s}$) and $\\sim 17,818\\times$ faster than a Cloud resync ($2.6835\\,\\text{s}$), enabling sub-millisecond physical disturbance handling without halting the production line.\n"
        "- **Pure Forward Pass Acceleration:** Pure ONNX policy inference ($0.0132\\,\\text{ms}$) achieves a $\\sim 321,152\\times$ ($\\sim 3.20 \\times 10^5\\times$) speedup over the Cloud solver.\n"
        "- **Full-Episode Autonomous Rescheduling:** Autonomous edge shift execution ($7.955\\,\\text{ms}$) is $\\sim 533\\times$ faster than Cloud NSGA-III optimization ($4.2392\\,\\text{s}$), enabling edge agents to re-plan local production orders rapidly during Cloud network disconnections.\n\n"
        "#### Trade-Off Analysis Across Operational Regimes (Figure 3):\n"
        "- **Mode 1** achieves low makespan ($65.35\\,\\text{min}$) and zero breakdowns ($0.00$), but incurs heavy communication chatter ($10$ Cloud calls, $220$ network packets) and $26.8\\,\\text{s}$ of accumulated solver wait latency.\n"
        "- **Mode 2** eliminates Cloud communication ($0$ calls, $0.0\\,\\text{s}$ solver delay), but suffers from uncorrected drift accumulation ($\\text{CDI}_{\\max} = 0.4255$, final $\\text{CDI} = 0.4193$), resulting in $2$ catastrophic tool breakdowns and severe makespan degradation ($83.93\\,\\text{min}$, $+30.4\\%$ longer than Mode 3).\n"
        "- **Mode 3 (Proposed)** filters stochastic sensor ripple and triggers Cloud re-optimization selectively exactly $2$ times (Op 20 and Op 38, during genuine bottleneck shocks on M4 and M3), reducing Cloud calls by **$80.0\\%$** ($2$ vs $10$) while maintaining a controlled drift trajectory ($\\text{CDI}_{\\text{final}} = 0.0548 < 0.12$). Mode 3 achieves the superior overall makespan ($64.37\\,\\text{min}$), lowest TOU cost ($\\$2.898$), and zero tool breakdowns ($0.00$), demonstrating the physical efficacy of the proposed digital twin.\n"
    )

    sec5_marker = "## 5. Mechatronic Hardware-in-the-Loop Telemetry"
    sec5_idx = content.find(sec5_marker)
    sec45_marker = "### 4.5 Closed-Loop Multi-Tier Latency"
    sec45_idx = content.find(sec45_marker)

    if sec45_idx != -1 and sec5_idx != -1:
        content = content[:sec45_idx] + sec45 + "\n---\n\n" + content[sec5_idx:]
    elif sec5_idx != -1:
        content = content[:sec5_idx] + "\n" + sec45 + "\n---\n\n" + content[sec5_idx:]

    with open('results/manuscript_draft.md', 'w') as f:
        f.write(content)

    print("Successfully updated manuscript_draft.md!")

if __name__ == '__main__':
    main()
