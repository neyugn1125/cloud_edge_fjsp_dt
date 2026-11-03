"""
Standalone Verification Script and Test Suite for CNC Machine Physical Twin.

Evaluates CNC machine M1 over 25 consecutive machining operations (t_base = 8.0 min)
under 3 operational scenarios:
    - Scenario A: Low cutting speed s=1 (v=1.0), no proactive PM.
    - Scenario B: High cutting speed s=3 (v=1.5), no proactive PM (suffers catastrophic breakdowns).
    - Scenario C: High cutting speed s=3 (v=1.5), with Opportunistic PM at wear >= 80% w_max.

Generates comparative telemetry metrics and exports verification plots to:
    tests/single_machine_verification.png
"""

from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend for automation
import matplotlib.pyplot as plt
import pytest

from sim_env.physical_machine import CNCPhysicalTwin, OperationExecutionResult
from utils.mqtt_schemas import MachineConfig, MachineState, load_system_config


def run_scenario(
    machine_config: MachineConfig,
    speed_level: int,
    pm_threshold_um: Optional[float] = None,
    num_operations: int = 25,
    t_base_min: float = 8.0,
    seed: int = 42,
) -> Tuple[CNCPhysicalTwin, List[OperationExecutionResult]]:
    """
    Execute a batch of operations on a physical machine twin under specified policy.

    Args:
        machine_config: MachineConfig instance.
        speed_level: Speed scaling level (1, 2, or 3).
        pm_threshold_um: Flank wear threshold to trigger proactive tool replacement.
        num_operations: Number of consecutive operations.
        t_base_min: Base cutting time per operation in minutes.
        seed: Random seed for stochastic wear perturbation.

    Returns:
        Tuple of (CNCPhysicalTwin, List[OperationExecutionResult]).
    """
    twin = CNCPhysicalTwin(
        machine_id="M1",
        config=machine_config,
        alpha_p=1.5,
        speed_levels={1: 1.0, 2: 1.25, 3: 1.5},
    )
    rng = random.Random(seed)
    results: List[OperationExecutionResult] = []

    for _ in range(num_operations):
        do_pm = False
        if pm_threshold_um is not None and twin.current_wear_um >= pm_threshold_um:
            do_pm = True

        res = twin.execute_operation(
            t_base_min=t_base_min,
            speed_level=speed_level,
            do_opportunistic_pm=do_pm,
            rng=rng,
        )
        results.append(res)

    return twin, results


def generate_verification_plot(
    results_a: List[OperationExecutionResult],
    results_b: List[OperationExecutionResult],
    results_c: List[OperationExecutionResult],
    w_max_um: float = 100.0,
    output_path: str = "tests/single_machine_verification.png",
) -> None:
    """
    Export 2-subplot publication-grade verification chart:
        1. Tool Flank Wear w_k (um) vs Operation Index.
        2. Real-time Cutting Power P_actual (kW) vs Operation Index.
    """
    ops_x = list(range(1, len(results_a) + 1))

    # Wear trajectories: record wear after each operation
    wear_a = [r.wear_after_um for r in results_a]
    wear_b = [r.wear_after_um for r in results_b]
    wear_c = [r.wear_after_um for r in results_c]

    # Power trajectories
    power_a = [r.p_actual_kw for r in results_a]
    power_b = [r.p_actual_kw for r in results_b]
    power_c = [r.p_actual_kw for r in results_c]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True, dpi=300)

    # --------------------------------------------------------------------------
    # Subplot 1: Tool Wear vs. Operation Index
    # --------------------------------------------------------------------------
    ax1.plot(ops_x, wear_a, "o-", color="#1f77b4", label="Scenario A: s=1 (v=1.0, No PM)", linewidth=1.8, markersize=5)
    ax1.plot(ops_x, wear_b, "s-", color="#d62728", label="Scenario B: s=3 (v=1.5, No PM)", linewidth=1.8, markersize=5)
    ax1.plot(ops_x, wear_c, "^-", color="#2ca02c", label="Scenario C: s=3 (v=1.5, PM @ 80 um)", linewidth=2.0, markersize=6)

    # Threshold guidelines
    ax1.axhline(y=w_max_um, color="black", linestyle="--", linewidth=1.4, label=rf"Max Wear Limit $w_{{max}}$ ({w_max_um:.0f} $\mu$m)")
    ax1.axhline(y=80.0, color="#ff7f0e", linestyle=":", linewidth=1.4, label=r"PM Threshold (80% $w_{max} = 80$ $\mu$m)")

    # Annotate breakdown events in Scenario B
    for idx, r in enumerate(results_b):
        if r.had_breakdown:
            ax1.scatter([idx + 1], [w_max_um], color="red", s=130, zorder=5, marker="X")
            ax1.annotate(
                f"Breakdown\n(+{r.t_breakdown_min:.0f}m penalty)",
                xy=(idx + 1, w_max_um),
                xytext=(idx - 2.5, w_max_um - 18),
                arrowprops=dict(facecolor="red", arrowstyle="->", lw=1.2),
                fontsize=8.5,
                fontweight="bold",
                color="red",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffe6e6", edgecolor="red", alpha=0.9),
            )

    # Annotate PM events in Scenario C
    for idx, r in enumerate(results_c):
        if r.had_pm:
            ax1.scatter([idx + 1], [0.0], color="green", s=100, zorder=5, marker="^")
            ax1.annotate(
                f"PM Replace\n(+{r.t_pm_min:.0f}m)",
                xy=(idx + 1, 0.0),
                xytext=(idx - 1.0, 15.0),
                arrowprops=dict(facecolor="green", arrowstyle="->", lw=1.2),
                fontsize=8.5,
                fontweight="bold",
                color="#006600",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#e6ffe6", edgecolor="green", alpha=0.9),
            )

    ax1.set_ylabel(r"Flank Tool Wear $VB$ ($\mu$m)", fontsize=11, fontweight="bold")
    ax1.set_ylim(-5, 115)
    ax1.set_title("Machine M1 Tool Flank Wear ($VB$) Trajectory across 25 Operations", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # --------------------------------------------------------------------------
    # Subplot 2: Actual Cutting Power with Tool Wear Drift
    # --------------------------------------------------------------------------
    ax2.plot(ops_x, power_a, "o-", color="#1f77b4", label="Scenario A: s=1 (v=1.0)", linewidth=1.8, markersize=5)
    ax2.plot(ops_x, power_b, "s-", color="#d62728", label="Scenario B: s=3 (v=1.5)", linewidth=1.8, markersize=5)
    ax2.plot(ops_x, power_c, "^-", color="#2ca02c", label="Scenario C: s=3 (PM @ 80 um)", linewidth=2.0, markersize=6)

    ax2.set_xlabel("Operation Sequence Index (1 .. 25)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cutting Power $P_{actual}$ (kW)", fontsize=11, fontweight="bold")
    ax2.set_title("Actual Cutting Power Drift $P_{actual}$ due to Flank Wear & Speed Scaling", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


# ==============================================================================
# Pytest Unit Tests
# ==============================================================================

@pytest.fixture
def m1_config() -> MachineConfig:
    sys_cfg = load_system_config("config/system_params.yaml")
    return sys_cfg.get_machine("M1")


def test_idle_gap_transitions(m1_config: MachineConfig):
    """
    Test evaluation of idle gaps:
        - gap <= 2.0 min: IDLE
        - 2.0 < gap < T_BE (9.0 min): STANDBY
        - gap >= T_BE: TURN_OFF
    """
    twin = CNCPhysicalTwin("M1", m1_config)
    assert twin.break_even_time_min == 9.0

    # Case 1: Short gap (1.5 min) -> IDLE
    state, energy = twin.evaluate_idle_gap(1.5)
    assert state == "IDLE"
    assert pytest.approx(energy, rel=1e-4) == 1.20 * (1.5 / 60.0)

    # Case 2: Intermediate gap (5.0 min) -> STANDBY
    state, energy = twin.evaluate_idle_gap(5.0)
    assert state == "STANDBY"
    assert pytest.approx(energy, rel=1e-4) == 0.20 * (5.0 / 60.0)

    # Case 3: Long gap (15.0 min >= T_BE) -> TURN_OFF
    state, energy = twin.evaluate_idle_gap(15.0)
    assert state == "TURN_OFF"
    assert pytest.approx(energy, rel=1e-4) == 0.15  # e_on_off_kwh

    # Test apply_idle_gap state update
    twin.apply_idle_gap(1.5)
    assert twin.state == MachineState.IDLE
    assert pytest.approx(twin.e_idle_kwh, rel=1e-4) == 1.20 * (1.5 / 60.0)

    twin.apply_idle_gap(5.0)
    assert twin.state == MachineState.STANDBY
    assert pytest.approx(twin.e_stdby_kwh, rel=1e-4) == 0.20 * (5.0 / 60.0)

    twin.apply_idle_gap(12.0)
    assert twin.state == MachineState.OFF
    assert pytest.approx(twin.e_on_off_kwh, rel=1e-4) == 0.15


def test_speed_scaling_and_taylor_wear(m1_config: MachineConfig):
    """
    Verify speed-scaling processing time, Taylor wear increment, and power drift.
    """
    twin = CNCPhysicalTwin("M1", m1_config)
    twin.reset(initial_wear_um=0.0)

    # Operation 1: s=1 (v=1.0)
    t_proc1 = 8.0 / 1.0
    expected_dw1 = twin.eta * (1.0 ** twin.beta) * t_proc1
    wear_avg1 = 0.5 * expected_dw1
    expected_p1 = twin.p_idle_kw + twin.p_cut_base_kw * (1.0 ** twin.alpha_p) * (1.0 + twin.gamma * (wear_avg1 / twin.w_max_um))
    r1 = twin.execute_operation(t_base_min=8.0, speed_level=1, rng=None)
    assert r1.t_proc_min == t_proc1
    assert pytest.approx(r1.delta_wear_um, rel=1e-4) == expected_dw1
    assert pytest.approx(r1.p_actual_kw, rel=1e-4) == expected_p1
    assert pytest.approx(r1.wear_after_um, rel=1e-4) == expected_dw1

    # Operation 2: s=3 (v=1.5)
    t_proc2 = 8.0 / 1.5
    expected_dw2 = twin.eta * (1.5 ** twin.beta) * t_proc2
    r2 = twin.execute_operation(t_base_min=8.0, speed_level=3, rng=None)
    assert pytest.approx(r2.t_proc_min, rel=1e-4) == t_proc2
    assert pytest.approx(r2.delta_wear_um, rel=1e-3) == expected_dw2
    assert pytest.approx(r2.wear_after_um, rel=1e-3) == expected_dw1 + expected_dw2


def test_scenario_comparison_assertions(m1_config: MachineConfig):
    """
    Verify the quantitative relationships between Scenarios A, B, and C.
    """
    twin_a, res_a = run_scenario(m1_config, speed_level=1, pm_threshold_um=None)
    twin_b, res_b = run_scenario(m1_config, speed_level=3, pm_threshold_um=None)
    twin_c, res_c = run_scenario(m1_config, speed_level=3, pm_threshold_um=80.0)

    # Scenario A: 25 ops with initial wear 55 um -> 2 breakdowns
    assert twin_a.num_breakdowns == 2
    assert twin_a.num_pm_replacements == 0
    assert pytest.approx(twin_a.available_time, rel=1e-3) == 240.0

    # Scenario B: High speed -> 4 breakdowns
    assert twin_b.num_breakdowns == 4
    assert twin_b.num_pm_replacements == 0
    assert pytest.approx(twin_b.available_time, rel=1e-3) == 213.3333

    # Scenario C: Opportunistic PM @ 80 um -> 0 breakdowns, 4 PMs
    assert twin_c.num_breakdowns == 0
    assert twin_c.num_pm_replacements == 4
    assert pytest.approx(twin_c.available_time, rel=1e-3) == 153.3333

    # Scenario C dominates Scenario B: Lower time AND lower energy
    assert twin_c.available_time < twin_b.available_time
    assert twin_c.total_energy_kwh < twin_b.total_energy_kwh


if __name__ == "__main__":
    sys_config = load_system_config("config/system_params.yaml")
    m1 = sys_config.get_machine("M1")

    print("=" * 82)
    print("CNC Physical Twin Single-Machine Verification (M1 - 25 Operations)")
    print("=" * 82)

    twin_a, res_a = run_scenario(m1, speed_level=1, pm_threshold_um=None)
    twin_b, res_b = run_scenario(m1, speed_level=3, pm_threshold_um=None)
    twin_c, res_c = run_scenario(m1, speed_level=3, pm_threshold_um=80.0)

    chart_path = "tests/single_machine_verification.png"
    generate_verification_plot(res_a, res_b, res_c, w_max_um=m1.tool_wear.w_max_um, output_path=chart_path)

    print(f"\n{'Metric':<32} | {'Scenario A (s=1)':<16} | {'Scenario B (s=3)':<16} | {'Scenario C (s=3+PM)':<16}")
    print("-" * 86)
    print(f"{'Cutting Speed Level':<32} | {'s=1 (v=1.0)':<16} | {'s=3 (v=1.5)':<16} | {'s=3 (v=1.5)':<16}")
    print(f"{'PM Policy':<32} | {'None':<16} | {'None':<16} | {'Wear >= 80 um':<16}")
    print(f"{'Total Completion Time (min)':<32} | {twin_a.available_time:<16.2f} | {twin_b.available_time:<16.2f} | {twin_c.available_time:<16.2f}")
    print(f"{'Total Energy Consumed (kWh)':<32} | {twin_a.total_energy_kwh:<16.2f} | {twin_b.total_energy_kwh:<16.2f} | {twin_c.total_energy_kwh:<16.2f}")
    print(f"{'Machining Energy E_proc (kWh)':<32} | {twin_a.e_proc_kwh:<16.2f} | {twin_b.e_proc_kwh:<16.2f} | {twin_c.e_proc_kwh:<16.2f}")
    print(f"{'Standby/Maint Energy (kWh)':<32} | {twin_a.e_stdby_kwh:<16.2f} | {twin_b.e_stdby_kwh:<16.2f} | {twin_c.e_stdby_kwh:<16.2f}")
    print(f"{'Proactive Replacements (PM)':<32} | {twin_a.num_pm_replacements:<16} | {twin_b.num_pm_replacements:<16} | {twin_c.num_pm_replacements:<16}")
    print(f"{'Catastrophic Breakdowns':<32} | {twin_a.num_breakdowns:<16} | {twin_b.num_breakdowns:<16} | {twin_c.num_breakdowns:<16}")
    print(f"{'Final Flank Wear VB (um)':<32} | {twin_a.current_wear_um:<16.2f} | {twin_b.current_wear_um:<16.2f} | {twin_c.current_wear_um:<16.2f}")
    print("-" * 86)

    print(f"\n[OK] Verification chart successfully generated: {chart_path}")
    print("=" * 82)
