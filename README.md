# Cloud-Edge Collaborative Digital Twin for Dynamic Green Flexible Job Shop Scheduling with Speed-Scaling and Tool Wear Resilience

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04%20%7C%2024.04%20LTS-E95420.svg?style=flat-square&logo=ubuntu&logoColor=white)](https://ubuntu.com/)
[![Firmware](https://img.shields.io/badge/Firmware-PlatformIO%20(ESP32)-FF7F00.svg?style=flat-square&logo=platformio&logoColor=white)](https://platformio.org/)
[![Edge AI](https://img.shields.io/badge/Edge%20AI-ONNX%20Runtime%20(CPU)-0078D4.svg?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Solver](https://img.shields.io/badge/Solver-Pymoo%20NSGA--III%20%26%20SB3%20MaskablePPO-blueviolet.svg?style=flat-square)](https://pymoo.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(Pytest%20Suite)-success.svg?style=flat-square)](#reproducibility-guide)

---

## 1. Executive Summary

This repository hosts the official source code, hardware-in-the-loop (HIL) telemetry, firmware, and reproducible experimental artifacts for the **Cloud-Edge Collaborative Digital Twin (DT)** framework designed for **Dynamic Green Flexible Job Shop Scheduling Problems (D-GFJSP)** with physical speed-scaling, real-time tool wear resilience, and dynamic disturbance rejection.

### The Industrial Problem
Modern discrete manufacturing systems (such as CNC machining centers) face multifaceted operating uncertainties:
1. **Tool Degradation & Micro-Stoppages:** Cutting tool flank wear progresses non-linearly, causing cutting forces, friction, and tool temperatures to rise. This escalates active machining power by up to $+28\%$ and drastically raises tool breakage risks if Condition-Based Maintenance (CBM) is misjudged.
2. **Volatile Energy Economics:** Industrial Time-of-Use (TOU) electricity pricing imposes steep peak-hour tariffs, necessitating intelligent load shifting and speed-scaling ($s \in \{1, 2, 3\}$).
3. **Latency & Computational Bottlenecks:** Centralized Cloud meta-heuristics (e.g., NSGA-III) require $4.24 \pm 0.03\,\text{s}$ on Mk01 ($14.97\,\text{s}$ on Mk03) to compute global Pareto schedules, rendering them incapable of reacting to sub-second shop-floor disruptions. Conversely, pure Edge reactive heuristics or dispatching rules suffer from localized myopia, degrading global factory efficiency ($C_{\max}$, tardiness).

### The Proposed Collaborative Digital Twin
Our framework resolves this trade-off via a tri-tier hierarchical architecture:
- **Cloud Macro-Optimizer:** Executes multi-objective reference-point evolutionary optimization (NSGA-III) across three competing objectives: Makespan ($C_{\max}$), TOU Electricity Cost ($\$$), and Total Weighted Tardiness ($\text{TWT}$).
- **Edge Micro-Controller:** Executes a 256-128 MLP policy trained via Maskable Proximal Policy Optimization (MaskablePPO) and exported to ONNX Runtime. The edge node performs real-time dispatching, dynamic cutting speed-scaling, and proactive tool replacements in **$0.0132\,\text{ms}$** forward pass and **$0.1506\,\text{ms}$** per decision step.
- **Event-Driven Synchronization via CDI:** Implements a shift-normalized **Cumulative Disturbance Index ($\text{CDI}(t)$)** that monitors factory makespan and energy deviations. Re-optimization requests are transmitted to the Cloud only when $\text{CDI}(t)$ exceeds the optimal threshold ($\theta_{\text{th}} = 0.12$), slashing network bandwidth and redundant solver calls by $80.0\%$.
- **Mechatronics & Hardware-in-the-Loop:** Integrates physical ESP32 dual-sensor telemetry (INA226 I2C current/power sensor, ADXL345 I2C vibration sensor, solid-state relays for zero-loss standby transitions) and bidirectional serial communication with GRBL v1.1 CNC hardware.

```mermaid
flowchart TD
    subgraph CloudTier["Cloud Tier (Macro-Optimization & Global Coordination)"]
        direction TB
        C1["Global Multi-Objective Solver<br/>(NSGA-III / MOEA-D, 100 Gens)"] -->|Pareto Baseline Schedule| C2["Cloud MQTT Broker & Twin Sync Engine"]
        C2 -->|Global Guidance Vector| E1
    end

    subgraph EdgeTier["Edge Tier (Micro-Control & Sub-Millisecond Decision Engine)"]
        direction TB
        E1["Edge Digital Twin Node<br/>(ONNX Runtime CPU Inference: 0.0132 ms)"] -->|Speed & Dispatch Actions| E2["GRBL Real-Time CNC Controller<br/>(Overrides: 100%, 125%, 150%)"]
        E3["CDI Calculation Engine<br/>(Shift-Normalized Makespan + Energy Drift)"] -->|Event Trigger: CDI > 0.12| C2
        E4["Dual-Sensor Feature Fusion & Estimator<br/>(INA226 Power + ADXL345 Vibration)"] -->|Wear Estimate: R^2 = 0.9984 (active) / 0.9951 (full)| E1
        E4 --> E3
    end

    subgraph PhysicalTier["Physical / Hardware-in-the-Loop Tier (Mechatronics & CNC Machine)"]
        direction TB
        P1["Physical CNC / Milling Workcell<br/>(GRBL v1.1, Real Serial 115200 bps)"] -->|Physical Cutting Dynamics| P2["Dual Sensor Suite<br/>(INA226 Power @ 0x40 + ADXL345 Vib @ 0x53)"]
        P2 -->|200 Hz Raw Telemetry| P3["ESP32 Microcontroller<br/>(Kalman Filter + 20-tap Sliding RMS)"]
        P3 -->|10 Hz JSON Telemetry via USB/UART| E4
        E2 -->|G-code & Real-Time Byte Overrides| P1
        E1 -->|SSR Standby Power Cutoff| P4["Opto-Isolated Solid State Relays<br/>(GPIO 25 Standby, GPIO 26 Spindle)"]
    end
```

---

## 2. Quantitative Highlights

All metrics reported below strictly reflect audited experimental runs across 15 independent test seeds (`TEST_SEEDS = [301..315]`):

| Category | Audited Metric | Empirical Observation & Contrast | Impact / Advantage |
| :--- | :--- | :--- | :--- |
| **Inference Latency** | **$0.0132\,\text{ms}$** | Edge ONNX forward policy inference ($0.0132 \pm 0.0013\,\text{ms}$; $0.1506 \pm 0.0064\,\text{ms}$ full decision step) vs. Cloud NSGA-III ($4.24 \pm 0.03\,\text{s}$) | **$\sim 2.81 \times 10^4 \times$ (per-step) / $\sim 3.20 \times 10^5 \times$ (forward) speedup** over Cloud solver; enables hard real-time dispatching. |
| **Reliability & Breakdowns** | **$0.00$ Failures** | $0.00$ breakdowns across all 5 Brandimarte benchmarks (Mk01–Mk05) vs. $0.33 - 12.67$ (open-loop metaheuristics) and $0.80 - 7.60$ (wear-blind Variant 1) | Eliminates catastrophic tool breakage through closed-loop sensing ($R^2 = 0.9984$ active cutting / $0.9951$ full 60s profile). |
| **Energy & Cost Savings** | **$30.95\%$ Reduction** | SSR-actuated standby mode reduces TOU electricity cost from $\$20.106$ to $\$13.883$ on Mk03 ($-30.95\%$ reduction, saving $\$6.223$ per shift; or $+44.82\%$ cost penalty if omitted) | Eliminates parasitic idling power during inter-operational waiting times. |
| **Pareto Quality** | **$\overline{\text{HV}} = 0.4695$** | Proposed DT achieves mean $\overline{\text{HV}}=0.4695$ and $\overline{\text{IGD}}=0.3796$ across all instances (Mk01–Mk05) with $0.00$ breakdowns | Statistically outperforms all 4 open-loop metaheuristics on Mk05 ($p_{\text{holm}} \le 0.0011$) and beats NSGA-III/NSGA-II on Mk03; saves $9.28\%$ TOU cost on Mk05 vs. Right-Shift with zero failures. |

---

## 3. Mathematical Foundations & Physical Coupling

### 3.1 Multi-Objective Optimization Problem
The shop-floor scheduling formulation targets three competing objectives:

$$\min \mathbf{F} = [f_1, f_2, f_3]$$

1. **Makespan ($f_1 = C_{\max}$):**
   $$C_{\max} = \max_{j \in J} C_j$$

2. **Total TOU Electricity Cost ($f_2$):**
   $$f_2 = \sum_{t=0}^{C_{\max}} C_{\text{TOU}}(t) \cdot P_{\text{total}}(t) \cdot \frac{\Delta t}{60} + \lambda_{\text{peak}} \sum_{t=0}^{C_{\max}} \max\left(0, P_{\text{total}}(t) - P_{\text{limit}}\right) \cdot \frac{\Delta t}{60}$$
   where $C_{\text{TOU}}(t)$ is the cyclic three-tier TOU tariff over repeating 100-minute manufacturing shifts:
   - **Off-Peak** $[0, 25)\,\text{min}$: $\$0.06/\text{kWh}$
   - **On-Peak** $[25, 60)\,\text{min}$: $\$0.20/\text{kWh}$
   - **Mid-Peak** $[60, 100)\,\text{min}$: $\$0.12/\text{kWh}$
   $P_{\text{total}}(t) = \sum_{k=1}^m P_k(t)$ is total plant power, $\lambda_{\text{peak}} = \$0.30/\text{kWh}$ is the peak demand excess surcharge, and $P_{\text{limit}} \in \{22.0, 24.0, 32.0, 30.0, 16.0\}\,\text{kW}$ is the instance-specific power ceiling across Brandimarte benchmarks `Mk01`–`Mk05`.

3. **Total Weighted Tardiness ($f_3 = \text{TWT}$):**
   $$\text{TWT} = \sum_{j \in J} w_j \cdot \max(0, C_j - d_j)$$
   where $w_j$ represents job priority weight and $d_j$ is the due date.

### 3.2 Dynamic Cutting Power Drift & Non-Linear Tool Deterioration Model
Physical flank wear $w_k(t)$ progresses non-linearly under the extended Taylor tool wear model across discrete cutting speeds $s \in \{1, 2, 3\}$ ($v_1 = 1.0, v_2 = 1.25, v_3 = 1.50$):

$$\Delta w = \eta_k \cdot (v_s)^{\beta_k} \cdot t_{\text{proc}} \cdot \xi_{\text{noise}}$$

where $\eta_k \in [0.95, 1.35]\,\mu\text{m/min}$ is the base wear rate, $\beta_k \in [2.20, 2.70]$ is the machine-specific Taylor velocity exponent across CNC workstations `M1`–`M8` in `config/system_params.yaml`, and $\xi_{\text{noise}} \sim \mathcal{N}(1.0, 0.03^2)$ captures metallurgical stochasticity.

As the cutting edge dulls, cutting friction and spindle loads increase active machining power:

$$P_{\text{cutting}, k}(v, w) = P_{\text{idle}, k} + P_{\text{cut\_base}, k} \cdot v^{\alpha_p} \cdot \left( 1 + \gamma_k \cdot \frac{w_k(t)}{w_{\max}} \right)$$

where $\alpha_p = 1.5$ is the cutting speed power-law scaling exponent, $\gamma_k \in [0.20, 0.28]$ (plant mean $0.25$) accounts for up to $+28\%$ power drift at terminal tool wear ($w_{\max} = 100.0\,\mu\text{m}$), and $w_k(t)$ is estimated continuously by the dual-sensor EKF fusion model.

### 3.3 Standby Power Transition & Break-Even Time ($T_{\text{BE}}$)
To eliminate energy waste during inter-operational idle intervals, machines transition to an ultra-low power standby state via opto-isolated solid-state relays (SSR) if the expected waiting duration $\Delta t$ exceeds the physical break-even threshold:

$$T_{\text{BE}} = \frac{E_{\text{on\_off}}}{P_{\text{idle}} - P_{\text{standby}}}$$

The framework characterizes break-even economics across two complementary physical scales:
1. **Industrial Shop-Floor CNC Centers (`M1`–`M8` in `config/system_params.yaml`):**
   - **Idle Power:** $P_{\text{idle}, k} \in [0.85, 1.50]\,\text{kW}$ ($1.00 - 1.50\,\text{kW}$ on heavy machining centers).
   - **Standby Power:** $P_{\text{stdby}, k} \in [0.15, 0.25]\,\text{kW}$ (auxiliary electronics and edge sensing retained).
   - **Transition / Restart Energy:** $E_{\text{on\_off}, k} \in [0.12, 0.22]\,\text{kWh}$ with warmup duration $t_{\text{warmup}} \in [2.0, 4.0]\,\text{min}$ and tool replacement duration $t_{\text{pm}} = 10.0\,\text{min}$ ($5.0\,\text{min}$ base).
   - **Break-Even Threshold Range:** $T_{\text{BE}, k} \in [8.36, 10.54]\,\text{min}$ (e.g., $9.00\,\text{min}$ on M1, $8.64\,\text{min}$ on M2, $10.54\,\text{min}$ on M3).
2. **Physical Breadboard Hardware Proxy (24V CNC Mechatronics):**
   - **Idle Power:** $P_{\text{idle}} = 15.6\,\text{W}$
   - **Standby Power:** $P_{\text{standby}} = 3.6\,\text{W}$
   - **Transition Energy:** $E_{\text{trans}} = 0.62\,\text{Wh}$ ($2232\,\text{J}$)
   - **Resulting Break-Even Threshold:** $T_{\text{BE}} \approx 3.1\,\text{min}$. If idle interval $\Delta t \le T_{\text{BE}}$, the machine stays in idle to avoid transition energy and wear penalties; otherwise, SSR switches to standby.

### 3.4 Shift-Normalized Cumulative Disturbance Index ($\text{CDI}(t)$)
To prevent network saturation and avoid computational re-optimization overhead from transient noise, the edge twin tracks deviation normalized against the shift planned baseline:

$$\text{CDI}(t) = \omega_c \frac{|C_{\text{actual}}(t) - C_{\text{base}}(t)|}{C_{\text{base}}^{\text{total}}} + \omega_e \frac{|E_{\text{actual}}(t) - E_{\text{base}}(t)|}{E_{\text{base}}^{\text{total}}}$$

- **Weighting Factors:** $\omega_c = 0.50$, $\omega_e = 0.50$.
- **Normalization Basis:** $C_{\text{base}}^{\text{total}}$ and $E_{\text{base}}^{\text{total}}$ are the total planned makespan and energy for the shift, eliminating numerical divergence at $t \to 0$.
- **Trigger Condition:** When $\text{CDI}(t) > \theta_{\text{th}} = 0.12$, an event-triggered MQTT message (`dt/fjsp/edge/trigger_reschedule`, QoS 1) prompts the Cloud solver to re-optimize remaining operations from the current factory snapshot.

---

## 4. Repository Layout

```
cloud_edge_fjsp_dt/
├── cloud_layer/                     # Cloud-tier macro-optimization & coordination
│   ├── nsga3_optimizer.py           # Multi-objective NSGA-III reference-direction solver
│   ├── generate_moead_baselines.py  # MOEA/D reference front baseline generator
│   ├── generate_multi_seed_baselines.py # Multi-seed independent solver baseline generator
│   └── cloud_mqtt_server.py         # Cloud twin synchronization & reschedule handler
├── edge_layer/                      # Edge-tier micro-control & real-time dispatching
│   ├── edge_twin_node.py            # Local DT node, CDI tracker & ONNX execution loop
│   ├── ppo_agent.py                 # MaskablePPO agent implementation (Stable-Baselines3)
│   ├── onnx_exporter.py             # PyTorch-to-ONNX compilation with validation checks
│   ├── grbl_controller.py           # GRBL CNC driver with real-time speed overrides
│   ├── breadboard_daq.py            # Dual-sensor hardware bridge & serial telemetry decoder
│   └── models/                      # Trained models & compiled ONNX artifacts
├── sim_env/                         # High-fidelity physics-informed simulation environment
│   ├── fjsp_gym_env.py              # Gymnasium-compatible 87-dim observation FJSP environment
│   ├── physical_machine.py          # Machine power states, tool wear drift & CBM model
│   └── run_baselines.py             # Heuristic rule dispatchers (SPT, EET, EDD, Random)
├── firmware_esp32/                  # Physical twin microcontroller firmware (ESP32)
│   ├── platformio.ini               # PlatformIO configuration (Espressif32, ArduinoJson)
│   ├── main.cpp                     # 200 Hz I2C sampling, 1D Kalman filter, 10 Hz JSON telemetry
│   └── README.md                    # Hardware wiring and flashing instructions
├── data/                            # Benchmark instances & experimental records
│   ├── benchmark_parser.py          # Brandimarte parser with green energy attributes
│   └── benchmarks/                  # Canonical datasets & reproducibility specifications
│       ├── Mk01_green.json ... Mk05_green.json
│       ├── phase1_baseline_results.csv
│       ├── phase2_comparison_results.csv
│       ├── phase3_hardware_telemetry.csv
│       ├── phase4_hil_metrics.csv
│       ├── phase5_wilcoxon_benchmark_table.csv
│       ├── phase5_ablation_results.csv
│       ├── phase5_cdi_sensitivity.csv
│       ├── multi_seed_schedules.json    # 15-seed independent optimization populations
│       ├── hv_normalization_bounds.json # Adaptive Ideal/Nadir bounds & reference point
│       └── reproducibility_specs.md # Single Source of Truth for experimental protocol
├── utils/                           # Core utilities, schemas, and metrics
│   ├── eval_protocol.py             # Disjoint seed sets, CDI calculation & audit contracts
│   ├── mqtt_broker.py               # Embedded local MQTT broker for headless testing
│   ├── mqtt_schemas.py              # Telemetry & control JSON schemas
│   └── stat_metrics.py              # HV, IGD, Wilcoxon test, Cliff's Delta, Holm-Bonferroni
├── tests/                           # Complete verification and evaluation suite (143 tests)
│   ├── test_task5_6b_stats_and_ablation.py # Multi-seed variance, Cliff's delta & ablation
│   ├── test_task5_5d_final_audit.py # Master verification of 7-point data consistency
│   ├── test_task4_3_full_hil.py     # HIL closed-loop integration test
│   ├── test_task5_2_multi_instance_eval.py
│   ├── test_task5_3_ablation_sensitivity.py
│   └── *.png                        # Publication-quality result figures (300 DPI)
├── config/                          # Centralized system parameters
│   └── system_params.yaml           # TOU tariffs, machine powers, wear thresholds
├── pytest.ini                       # Pytest test discovery & warning filters
├── requirements.txt                 # Pinned Python package dependencies
└── CITATION.cff                     # Citation Metadata (CFF v1.2.0)
```

---

## 5. Hardware-in-the-Loop Setup & Pinout

The physical twin utilizes an **ESP32 DevKitC** interfacing with industrial sensing and actuation circuitry over a dedicated $400\,\text{kHz}$ Fast-Mode I2C bus and opto-isolated solid-state relays:

| Peripheral | Component | Physical Pin (ESP32) | Protocol / Interface | Function / Specification |
| :--- | :--- | :--- | :--- | :--- |
| **Current / Power Sensor** | Texas Instruments INA226 | `GPIO 21` (SDA)<br/>`GPIO 22` (SCL) | I2C (`0x40`, $400\,\text{kHz}$) | $10\,\text{m}\Omega$ shunt resistor, measures $V_{\text{bus}}$ ($1.25\,\text{mV/LSB}$) and $I_{\text{shunt}}$ ($2.5\,\mu\text{V/LSB}$). |
| **Vibration Accelerometer** | Analog Devices ADXL345 | `GPIO 21` (SDA)<br/>`GPIO 22` (SCL) | I2C (`0x53`, $400\,\text{kHz}$) | Full-resolution $\pm 2\,\text{g}$ ($3.9\,\text{mg/LSB}$), $20$-sample sliding RMS calculation. |
| **Standby Power Actuator** | Opto-isolated SSR Module | `GPIO 25` | Digital Output ($3.3\,\text{V}$) | Cut off auxiliary control power during machine standby. |
| **Spindle / Main Power** | Opto-isolated SSR Module | `GPIO 26` | Digital Output ($3.3\,\text{V}$) | Main machine tool power interlock. |
| **Telemetry Interface** | USB-UART Bridge | `GPIO 1` (TX0)<br/>`GPIO 3` (RX0) | UART ($115200\,\text{bps}$) | $10\,\text{Hz}$ JSON-serialized physical telemetry stream. |
| **CNC Machine Controller** | GRBL v1.1 Workcell | USB / Serial Port | UART ($115200\,\text{bps}$) | G-code streaming and real-time speed override injection. |

### GRBL v1.1 Real-Time Speed Override Protocol
Speed scaling commands chosen by the Edge ONNX policy ($s \in \{1, 2, 3\}$) are transmitted directly to the GRBL controller via real-time byte sequences:
- **Level 1 ($s=1$, Nominal $v=1.00$):** Reset commands `0x90` (Feed 100%) and `0x99` (Spindle 100%).
- **Level 2 ($s=2$, Fast $v=1.25$):** Reset to 100% (`0x90`, `0x99`), followed by two `+10%` (`0x91`, `0x9A`) and five `+1%` (`0x93`, `0x9C`) steps for Feed and Spindle.
- **Level 3 ($s=3$, Maximum $v=1.50$):** Reset to 100% (`0x90`, `0x99`), followed by five `+10%` (`0x91`, `0x9A`) steps for Feed and Spindle.

---

## 6. Reproducibility Guide

To ensure zero circular validation or data snooping, all evaluations adhere to the following **Disjoint Seed Protocol** defined in `utils/eval_protocol.py`:
- **Training Seeds (`TRAIN_SEEDS`):** `[42, 101, 2026]` (MaskablePPO policy training).
- **Validation Seeds (`VAL_SEEDS`):** `[201..210]` (10 seeds, CDI threshold grid sweep).
- **Testing Seeds (`TEST_SEEDS`):** `[301..315]` (15 seeds, reported benchmark tables and figures).

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/neyugn/cloud_edge_fjsp_dt.git
cd cloud_edge_fjsp_dt

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: One-Click Test Suite & Data Audit Verification
To run all 143 automated verification tests and check numerical consistency against the master audit protocol:
```bash
pytest -q
```
*Expected result:* `143 passed in ~24s`.

### Step 3: Reproduce Key Experiments
```bash
# 1. Run Baseline Heuristics & Static Cloud Optimization (Phase 1 & 2)
python tests/test_baselines.py
python cloud_layer/nsga3_optimizer.py

# 2. Run Edge DRL Training & ONNX Export (Phase 2)
python edge_layer/ppo_agent.py
python edge_layer/onnx_exporter.py

# 3. Verify ESP32 Hardware DSP & Dual-Sensor Tool Wear Fusion (Phase 3)
python tests/test_task3_3_full_profile.py

# 4. Execute Closed-Loop HIL Telemetry & Synchronization (Phase 4)
python tests/test_task4_3_full_hil.py

# 5. Execute 15-Seed Multi-Instance Benchmarks (Phase 5.5B)
python tests/test_task5_2_multi_instance_eval.py

# 6. Execute Ablation Study & CDI Sensitivity Analysis (Phase 5.5C & 5.5D)
python tests/test_task5_3_ablation_sensitivity.py

# 7. Run Final 7-Point Audit Verification
python tests/test_task5_5d_final_audit.py
```

---

## 7. Key Experimental Results & Visual Gallery

### 7.1 Multi-Instance Benchmark Results (15 Independent Test Seeds)
Performance metrics across Brandimarte benchmark instances under stochastic wear disturbances ($+15\% \dots +35\%$), evaluated with independent seed optimization across 10 distinct dispatching and optimization paradigms (15 runs each, total 750 evaluations):

| Problem Instance | Method | Hypervolume (HV) $\uparrow$ | Inverted Generational Distance (IGD) $\downarrow$ | Makespan $C_{\max}$ (min) $\downarrow$ | TOU Cost (\$) $\downarrow$ | Tardiness TWT (min) $\downarrow$ | Breakdowns $\downarrow$ | Cliff's Delta ($\delta$) | $p_{\text{holm}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mk01**<br>(10x6, 55 ops) | Rule-SPT-Fast | $0.0890 \pm 0.0172$ | $0.8214 \pm 0.0431$ | $116.22 \pm 1.57$ | \$3.127 $\pm$ 0.118 | $550.7 \pm 27.2$ | 4.80 | +1.00 | < 0.001 |
|  | Rule-EET-Eco | $0.3441 \pm 0.0263$ | $0.4721 \pm 0.0268$ | $94.67 \pm 2.49$ | \$2.455 $\pm$ 0.043 | $424.1 \pm 11.3$ | 0.00 | +0.41 | 0.2549 |
|  | Rule-EDD-Bal | $0.0396 \pm 0.0227$ | $1.0095 \pm 0.1120$ | $124.55 \pm 8.38$ | \$3.662 $\pm$ 0.070 | $544.3 \pm 80.7$ | 0.00 | +1.00 | < 0.001 |
|  | Right-Shift Rescheduling | $0.5202 \pm 0.1227$ | $0.3219 \pm 0.0862$ | $79.58 \pm 5.95$ | \$2.594 $\pm$ 0.152 | $321.1 \pm 95.1$ | 0.00 | -0.48 | 0.0747 |
|  | NSGA-II (Open-Loop) | $0.4845 \pm 0.2341$ | $0.3919 \pm 0.1989$ | $82.69 \pm 11.90$ | \$2.573 $\pm$ 0.199 | $365.4 \pm 159.8$ | 0.73 | -0.21 | 0.5537 |
|  | MOEA-D (Open-Loop) | $0.5326 \pm 0.1997$ | $0.3389 \pm 0.1403$ | $79.77 \pm 10.15$ | \$2.606 $\pm$ 0.127 | $323.2 \pm 128.2$ | 0.80 | -0.38 | 0.3616 |
|  | RVEA (Open-Loop) | $0.6183 \pm 0.2444$ | $0.3748 \pm 0.1622$ | $67.47 \pm 10.34$ | \$3.052 $\pm$ 0.368 | $184.0 \pm 105.6$ | 0.60 | -0.62 | 0.1077 |
|  | NSGA-III (Open-Loop) | $0.4693 \pm 0.1798$ | $0.3855 \pm 0.1460$ | $83.64 \pm 11.90$ | \$2.570 $\pm$ 0.233 | $362.7 \pm 101.7$ | 0.87 | -0.21 | 0.5614 |
|  | Standalone Edge PPO (No Cloud Prior) | $0.0548 \pm 0.0179$ | $0.9142 \pm 0.0605$ | $110.42 \pm 3.77$ | \$3.245 $\pm$ 0.136 | $673.6 \pm 45.8$ | 0.00 | +1.00 | < 0.001 |
|  | **Proposed Cloud-Edge DT (NSGA-III + Edge ONNX + CDI)** | **0.4103 $\pm$ 0.1024** | **0.3872 $\pm$ 0.0951** | **86.00 $\pm$ 9.79** | **\$2.793 $\pm$ 0.183** | **330.0 $\pm$ 60.5** | **0.00** | Base | Base |
| **Mk02**<br>(10x6, 58 ops) | Rule-SPT-Fast | $0.1294 \pm 0.0761$ | $0.8712 \pm 0.1810$ | $110.98 \pm 16.66$ | \$3.323 $\pm$ 0.173 | $513.5 \pm 146.4$ | 5.67 | +1.00 | < 0.001 |
|  | Rule-EET-Eco | $0.1173 \pm 0.0331$ | $0.8847 \pm 0.0484$ | $98.73 \pm 2.46$ | \$1.962 $\pm$ 0.033 | $707.5 \pm 33.2$ | 0.00 | +1.00 | < 0.001 |
|  | Rule-EDD-Bal | $0.1385 \pm 0.0346$ | $0.8303 \pm 0.0593$ | $85.69 \pm 2.79$ | \$4.311 $\pm$ 0.240 | $304.2 \pm 17.2$ | 0.00 | +1.00 | < 0.001 |
|  | Right-Shift Rescheduling | $0.7780 \pm 0.1332$ | $0.2376 \pm 0.0559$ | $66.09 \pm 8.85$ | \$2.175 $\pm$ 0.217 | $162.1 \pm 61.1$ | 0.00 | -0.68 | 0.0015 |
|  | NSGA-II (Open-Loop) | $0.6466 \pm 0.1645$ | $0.3094 \pm 0.1190$ | $69.11 \pm 10.70$ | \$2.172 $\pm$ 0.173 | $265.9 \pm 102.0$ | 0.60 | -0.28 | 0.7571 |
|  | MOEA-D (Open-Loop) | $0.5733 \pm 0.1908$ | $0.3589 \pm 0.0957$ | $70.52 \pm 10.39$ | \$2.345 $\pm$ 0.271 | $294.8 \pm 99.5$ | 0.87 | +0.15 | 0.8904 |
|  | RVEA (Open-Loop) | $0.7062 \pm 0.2558$ | $0.3388 \pm 0.1298$ | $54.86 \pm 8.23$ | \$2.846 $\pm$ 0.445 | $157.9 \pm 97.8$ | 0.80 | -0.23 | 0.7571 |
|  | NSGA-III (Open-Loop) | $0.7691 \pm 0.1572$ | $0.2460 \pm 0.0652$ | $66.29 \pm 9.09$ | \$2.095 $\pm$ 0.162 | $188.2 \pm 84.1$ | 0.33 | -0.59 | 0.0269 |
|  | Standalone Edge PPO (No Cloud Prior) | $0.1471 \pm 0.1101$ | $0.8577 \pm 0.2092$ | $99.79 \pm 18.09$ | \$3.399 $\pm$ 0.251 | $527.3 \pm 207.5$ | 0.53 | +0.98 | < 0.001 |
|  | **Proposed Cloud-Edge DT (NSGA-III + Edge ONNX + CDI)** | **0.5828 $\pm$ 0.1517** | **0.3522 $\pm$ 0.1046** | **72.31 $\pm$ 10.53** | **\$2.466 $\pm$ 0.355** | **237.7 $\pm$ 110.6** | **0.00** | Base | Base |
| **Mk03**<br>(15x8, 150 ops) | Rule-SPT-Fast | $0.2342 \pm 0.0375$ | $0.6191 \pm 0.0516$ | $547.20 \pm 11.86$ | \$17.007 $\pm$ 0.472 | $5394.7 \pm 260.1$ | 19.20 | +0.79 | 0.0070 |
|  | Rule-EET-Eco | $0.4781 \pm 0.1232$ | $0.4316 \pm 0.1095$ | $512.67 \pm 32.41$ | \$13.323 $\pm$ 0.275 | $4542.0 \pm 474.9$ | 1.40 | +0.24 | 0.9084 |
|  | Rule-EDD-Bal | $0.0742 \pm 0.0437$ | $0.9526 \pm 0.0985$ | $619.12 \pm 48.85$ | \$26.912 $\pm$ 0.867 | $3079.4 \pm 433.4$ | 4.87 | +1.00 | < 0.001 |
|  | Right-Shift Rescheduling | $0.8077 \pm 0.1111$ | $0.2440 \pm 0.0542$ | $420.24 \pm 26.54$ | \$13.479 $\pm$ 0.435 | $3606.0 \pm 529.7$ | 0.07 | -0.72 | < 0.001 |
|  | NSGA-II (Open-Loop) | $0.3565 \pm 0.1265$ | $0.5288 \pm 0.1385$ | $507.85 \pm 47.47$ | \$14.506 $\pm$ 0.677 | $5220.6 \pm 644.7$ | 10.07 | +0.52 | 0.0410 |
|  | MOEA-D (Open-Loop) | $0.4270 \pm 0.1863$ | $0.4741 \pm 0.1836$ | $496.38 \pm 44.33$ | \$14.649 $\pm$ 0.390 | $4888.5 \pm 1040.2$ | 10.47 | +0.32 | 0.9084 |
|  | RVEA (Open-Loop) | $0.5625 \pm 0.1864$ | $0.3681 \pm 0.1345$ | $445.82 \pm 31.79$ | \$15.346 $\pm$ 0.670 | $4280.2 \pm 1047.0$ | 12.00 | -0.00 | 0.9084 |
|  | NSGA-III (Open-Loop) | $0.3436 \pm 0.1141$ | $0.5395 \pm 0.1339$ | $514.37 \pm 25.12$ | \$14.280 $\pm$ 0.517 | $5239.8 \pm 815.4$ | 9.67 | +0.57 | 0.0214 |
|  | Standalone Edge PPO (No Cloud Prior) | $0.0978 \pm 0.0450$ | $0.9073 \pm 0.0881$ | $438.62 \pm 44.32$ | \$26.469 $\pm$ 1.338 | $5093.4 \pm 591.9$ | 6.27 | +1.00 | < 0.001 |
|  | **Proposed Cloud-Edge DT (NSGA-III + Edge ONNX + CDI)** | **0.5445 $\pm$ 0.2087** | **0.3981 $\pm$ 0.1637** | **462.69 $\pm$ 45.99** | **\$13.883 $\pm$ 0.677** | **4621.8 $\pm$ 889.6** | **0.00** | Base | Base |
| **Mk04**<br>(15x8, 90 ops) | Rule-SPT-Fast | $0.0652 \pm 0.0240$ | $0.9040 \pm 0.0688$ | $188.05 \pm 10.17$ | \$6.811 $\pm$ 0.127 | $1942.4 \pm 97.4$ | 8.53 | +1.00 | < 0.001 |
|  | Rule-EET-Eco | $0.1464 \pm 0.0270$ | $0.6577 \pm 0.0449$ | $195.60 \pm 4.47$ | \$4.430 $\pm$ 0.141 | $2038.0 \pm 49.3$ | 0.00 | +0.96 | < 0.001 |
|  | Rule-EDD-Bal | $0.0833 \pm 0.0063$ | $0.8873 \pm 0.0154$ | $196.24 \pm 4.35$ | \$6.740 $\pm$ 0.145 | $1414.7 \pm 26.6$ | 0.00 | +1.00 | < 0.001 |
|  | Right-Shift Rescheduling | $0.5651 \pm 0.2029$ | $0.3359 \pm 0.1300$ | $148.54 \pm 12.80$ | \$4.475 $\pm$ 0.218 | $1501.2 \pm 420.8$ | 0.00 | -0.56 | 0.0214 |
|  | NSGA-II (Open-Loop) | $0.3237 \pm 0.1499$ | $0.4653 \pm 0.1457$ | $168.22 \pm 15.41$ | \$4.542 $\pm$ 0.382 | $1840.5 \pm 348.0$ | 2.53 | +0.25 | 0.4280 |
|  | MOEA-D (Open-Loop) | $0.4374 \pm 0.2159$ | $0.4118 \pm 0.1889$ | $162.66 \pm 23.47$ | \$4.660 $\pm$ 0.380 | $1596.2 \pm 279.5$ | 2.87 | -0.16 | 1.0000 |
|  | RVEA (Open-Loop) | $0.4047 \pm 0.2414$ | $0.5033 \pm 0.1725$ | $157.56 \pm 21.13$ | \$5.319 $\pm$ 0.569 | $1508.7 \pm 467.3$ | 3.47 | -0.01 | 1.0000 |
|  | NSGA-III (Open-Loop) | $0.3603 \pm 0.1830$ | $0.4421 \pm 0.1385$ | $166.18 \pm 15.65$ | \$4.462 $\pm$ 0.336 | $1784.7 \pm 373.7$ | 2.47 | +0.20 | 1.0000 |
|  | Standalone Edge PPO (No Cloud Prior) | $0.1945 \pm 0.1139$ | $0.7548 \pm 0.1364$ | $151.79 \pm 16.68$ | \$6.835 $\pm$ 0.451 | $1476.8 \pm 252.8$ | 0.80 | +0.68 | 0.0037 |
|  | **Proposed Cloud-Edge DT (NSGA-III + Edge ONNX + CDI)** | **0.3912 $\pm$ 0.1429** | **0.3986 $\pm$ 0.1034** | **161.25 $\pm$ 13.95** | **\$4.632 $\pm$ 0.264** | **1700.1 $\pm$ 268.0** | **0.00** | Base | Base |
| **Mk05**<br>(15x4, 106 ops) | Rule-SPT-Fast | $0.1587 \pm 0.0504$ | $0.6929 \pm 0.0711$ | $377.60 \pm 20.43$ | \$19.270 $\pm$ 0.987 | $4456.9 \pm 361.5$ | 17.13 | +0.96 | 0.0011 |
|  | Rule-EET-Eco | $0.2303 \pm 0.0599$ | $0.4975 \pm 0.0738$ | $453.73 \pm 18.80$ | \$10.512 $\pm$ 0.197 | $6619.3 \pm 355.2$ | 0.00 | +0.83 | 0.0017 |
|  | Rule-EDD-Bal | $0.5623 \pm 0.0425$ | $0.3946 \pm 0.0131$ | $387.35 \pm 10.46$ | \$14.239 $\pm$ 0.343 | $2829.1 \pm 82.3$ | 0.00 | -0.85 | 0.0011 |
|  | Right-Shift Rescheduling | $0.5406 \pm 0.1007$ | $0.3302 \pm 0.0283$ | $370.30 \pm 28.80$ | \$12.343 $\pm$ 0.783 | $4671.8 \pm 761.3$ | 0.07 | -0.56 | < 0.001 |
|  | NSGA-II (Open-Loop) | $0.2084 \pm 0.0637$ | $0.4845 \pm 0.0809$ | $459.37 \pm 25.44$ | \$12.362 $\pm$ 0.650 | $6215.3 \pm 664.6$ | 9.13 | +0.88 | < 0.001 |
|  | MOEA-D (Open-Loop) | $0.1467 \pm 0.1102$ | $0.6129 \pm 0.1798$ | $470.79 \pm 46.37$ | \$12.719 $\pm$ 1.261 | $6984.4 \pm 1206.9$ | 9.87 | +0.88 | < 0.001 |
|  | RVEA (Open-Loop) | $0.1992 \pm 0.0785$ | $0.5436 \pm 0.0893$ | $446.88 \pm 25.00$ | \$16.178 $\pm$ 0.840 | $5054.0 \pm 836.0$ | 12.67 | +0.88 | 0.0011 |
|  | NSGA-III (Open-Loop) | $0.2318 \pm 0.1186$ | $0.4873 \pm 0.1248$ | $450.84 \pm 40.10$ | \$12.697 $\pm$ 0.921 | $6100.4 \pm 983.7$ | 9.20 | +0.74 | 0.0011 |
|  | Standalone Edge PPO (No Cloud Prior) | $0.4012 \pm 0.0891$ | $0.4503 \pm 0.0410$ | $344.81 \pm 25.02$ | \$15.457 $\pm$ 0.502 | $4896.0 \pm 639.0$ | 4.73 | +0.16 | 0.5995 |
|  | **Proposed Cloud-Edge DT (NSGA-III + Edge ONNX + CDI)** | **0.4188 $\pm$ 0.1046** | **0.3619 $\pm$ 0.0715** | **408.76 $\pm$ 29.48** | **\$11.198 $\pm$ 0.539** | **5395.5 $\pm$ 800.5** | **0.00** | Base | Base |
*Overall across all 5 benchmark instances, the Proposed Cloud-Edge DT achieves a mean Hypervolume $\overline{\text{HV}} = 0.4695$ and mean $\overline{\text{IGD}} = 0.3796$, with exactly $0.00$ tool failures across all evaluated configurations.*

![Multi-Instance Hypervolume and IGD](tests/phase5_multi_instance_hv_igd.png)

---

### 7.2 Ablation Study (15 Test Seeds on Mk01 & Mk03)
Quantifying the isolated contribution of each architectural component under cyclic TOU tariff (\$0.06 $\to$ \$0.20 $\to$ \$0.12 / kWh):

| Problem Instance | Architectural Variant | Description & Mechanism | Makespan $C_{\max}$ (min) | TOU Cost (\$) | Tardiness TWT (min) | Tool Breakdowns |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Mk01**<br>(10x6, 55 ops) | **Full Proposed Cloud-Edge DT** | Full framework with Dual-Sensor Fusion, Speed Scaling, SSR Standby & Cloud Guidance | **86.00 $\pm$ 9.79** | **\$2.793 $\pm$ 0.183** | **329.99 $\pm$ 60.47** | **0.00** |
|  | Variant 1 (w/o Dual-Sensor Wear Feedback) | Blind to wear disturbance; nominal wear estimation causes delayed PM and tool breakdowns | $84.20 \pm 9.27$ (-2.1%) | \$2.572 $\pm$ 0.157 (**-7.9%**) | $373.68 \pm 129.18$ (+13.2%) | **0.80** |
|  | Variant 2 (Fixed Speed s=2) | Locked at nominal speed s=2; lacks dynamic TOU tariff peak avoidance | $79.34 \pm 6.61$ (-7.8%) | \$3.031 $\pm$ 0.357 (**+8.5%**) | $294.04 \pm 62.85$ (-10.9%) | 0.00 |
|  | Variant 3 (Idle-Only / No Standby) | Lacks SSR power switching; machine continuously draws IDLE power during waiting intervals | $85.09 \pm 8.56$ (-1.1%) | \$3.224 $\pm$ 0.214 (**+15.4%**) | $324.19 \pm 49.99$ (-1.8%) | 0.00 |
|  | Variant 4 (w/o Cloud Guidance) | Pure local edge dispatching without Cloud global macro optimization schedule | $110.42 \pm 3.77$ (+28.4%) | \$3.245 $\pm$ 0.136 (**+16.2%**) | $673.57 \pm 45.85$ (+104.1%) | 0.00 |
| **Mk03**<br>(15x8, 150 ops) | **Full Proposed Cloud-Edge DT** | Full framework with Dual-Sensor Fusion, Speed Scaling, SSR Standby & Cloud Guidance | **462.69 $\pm$ 45.99** | **\$13.883 $\pm$ 0.677** | **4621.78 $\pm$ 889.63** | **0.00** |
|  | Variant 1 (w/o Dual-Sensor Wear Feedback) | Blind to wear disturbance; nominal wear estimation causes delayed PM and tool breakdowns | $511.54 \pm 29.06$ (+10.6%) | \$13.683 $\pm$ 0.387 (**-1.4%**) | $5226.80 \pm 801.80$ (+13.1%) | **7.60** |
|  | Variant 2 (Fixed Speed s=2) | Locked at nominal speed s=2; lacks dynamic TOU tariff peak avoidance | $413.64 \pm 35.03$ (-10.6%) | \$14.133 $\pm$ 0.969 (**+1.8%**) | $3750.56 \pm 725.84$ (-18.9%) | 0.00 |
|  | Variant 3 (Idle-Only / No Standby) | Lacks SSR power switching; machine continuously draws IDLE power during waiting intervals | $442.95 \pm 33.98$ (-4.3%) | \$20.106 $\pm$ 0.934 (**+44.8%**) | $4275.74 \pm 682.96$ (-7.5%) | 0.00 |
|  | Variant 4 (w/o Cloud Guidance) | Pure local edge dispatching without Cloud global macro optimization schedule | $438.62 \pm 44.32$ (-5.2%) | \$26.469 $\pm$ 1.338 (**+90.7%**) | $5093.45 \pm 591.89$ (+10.2%) | **6.27** |
#### Key Component Insights from Ablation Study:
1. **Scale-Dependent Behavior of Variant 1 (w/o Wear Feedback):**
   - On **Mk01** (55 operations): Variant 1 suffers **$0.80$ breakdowns** (vs. $0.00$ for Full Proposed) and inflates TWT by $+13.24\%$ ($373.68 \pm 129.18\,\text{min}$ vs $329.99 \pm 60.47\,\text{min}$). Because the edge agent lacks live sensor feedback, it assumes nominal tool wear, delays preventive maintenance, and risks severe tooling damage under unmodeled hardness surges (the *Blind Closed-Loop Hazard*).
   - On **Mk03** (150 operations): Under heavy cumulative wear, Variant 1 experiences **$7.60$ breakdowns** per run (compared to $0.00$ for Full Proposed and $9.67$ for open-loop NSGA-III), driving makespan to $511.54 \pm 29.06\,\text{min}$ ($+10.56\%$) and TWT to $5226.80 \pm 801.80\,\text{min}$ ($+13.09\%$).
2. **Trade-Off Analysis of Variant 2 (Fixed Speed $s=2$):**
   - Locking spindle cutting speeds to nominal level $s=2$ represents an operational trade-off rather than an unmitigated defect: it shortens Makespan ($-7.75\%$ on Mk01 from $86.00$ to $79.34\,\text{min}$, $-10.60\%$ on Mk03 from $462.69$ to $413.64\,\text{min}$) and reduces TWT ($-10.89\%$ and $-18.85\%$) because machines never decelerate to conservative speed $s=1$.
   - However, Variant 2 loses the agility to dynamically throttle cutting during expensive On-Peak tariff windows ($[25..60]\,\text{min}$ at $\$0.20/\text{kWh}$), which increases TOU electricity cost by **$+8.53\%$** on Mk01 ($\$3.031 \pm 0.357$ vs $\$2.793 \pm 0.183$) and **$+1.80\%$** on Mk03 ($\$14.133 \pm 0.969$ vs $\$13.883 \pm 0.677$).
3. **Standby Actuation (Variant 3):**
   - Eliminating SSR standby gating forces machines to draw continuous idling power during starvation buffers, increasing TOU electricity cost by **$+15.44\%$** on Mk01 ($\$3.224$ vs $\$2.793$) and **$+44.82\%$** on Mk03 ($\$20.106$ vs $\$13.883$). Conversely, activating SSR standby gating achieves a **$13.37\%$** cost reduction on Mk01 (saving $\$0.431$ per shift) and a **$30.95\%$** reduction on Mk03 (saving $\$6.223$ per shift).

![Ablation Study and Sensitivity Analysis](tests/phase5_ablation_and_sensitivity.png)

---

### 7.3 Closed-Loop HIL Synchronization & CDI Sensitivity Analysis

Comparison of synchronization paradigms on Mk01 ($55$ operations) under simultaneous continuous wear disturbance ($+15\% \dots +30\%$) and dual macro-disturbances (Op 18..22 hardness surge, Op 36..40 machine micro-stoppage) evaluated across 15 independent test seeds (`TEST_SEEDS = [301..315]`):

| Control Paradigm | Synchronization Strategy | Cloud Calls | Total Cloud Overhead | Edge Latency | MQTT RTT | Makespan $C_{\max}$ | Energy (kWh) | TOU Cost (\$) | Max CDI | Failures |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mode 1** | Periodic Cloud Rescheduling (every 5 ops) | 10 | 26.8\,\text{s} | $0.0\,\text{ms}$ | $0.0814\,\text{ms}$ | 65.35\,\text{min} | 19.226 | \$2.919 | 0.0988 | 0 |
| **Mode 2** | One-Shot Cloud + Edge Autonomous (No Re-sync) | 0 ($-100\%$) | 0.0\,\text{s} | 0.0132\,\text{ms} | $0.0814\,\text{ms}$ | 83.93\,\text{min} | 21.026 | \$2.920 | 0.4255 | **2** |
| **Mode 3** | **Proposed Event-Driven DT ($\theta_{\text{th}} = 0.12$)** | **2 ($-80\%$)** | **5.4\,\text{s}** | **0.0132\,\text{ms}** | $0.0814\,\text{ms}$ | **64.37\,\text{min}** | **19.311** | **\$2.898** | **0.1924** | **0** |
![Closed-Loop HIL Real-Time Execution](tests/phase4_closed_loop_hil.png)

#### Full Parametric CDI Threshold Sensitivity Evaluation (10 Evaluation Thresholds, 10 Validation Seeds `VAL_SEEDS = [201..210]`)
Trade-off between Cloud re-optimization frequency, schedule degradation, and total operational cost index:
$$\text{Total\_System\_Cost\_Index} = 0.45 \frac{C_{\max}}{C_{\text{ref}}} + 0.45 \frac{\text{Cost}}{\text{Cost}_{\text{ref}}} + 0.10 \frac{\text{Cloud\_Calls}}{10.0}$$

| CDI Threshold $\theta_{\text{th}}$ | Cloud Calls ($\text{Mean} \pm \text{Std}$) | Makespan ($\text{Mean} \pm \text{Std}$) | TOU Cost (\$) | Max CDI ($\text{Mean} \pm \text{Std}$) | Schedule Index | Total System Cost Index | Operational Region |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.02** | $44.30 \pm 2.10$ | $55.93 \pm 2.68$ | \$3.346 \pm 0.116 | $0.1113 \pm 0.0066$ | 1.1667 | 1.4931 | Over-Triggering Region |
| **0.05** | $10.40 \pm 0.49$ | $58.46 \pm 1.50$ | \$3.138 \pm 0.111 | $0.1077 \pm 0.0123$ | 1.1461 | 1.1355 | Over-Triggering Region |
| **0.08** | $2.20 \pm 0.40$ | $67.52 \pm 6.91$ | \$2.805 \pm 0.138 | $0.1901 \pm 0.0199$ | 1.1565 | 1.0629 | Robust Operating Plateau |
| **0.10** | $2.10 \pm 0.30$ | $68.09 \pm 7.31$ | \$2.780 \pm 0.136 | $0.1923 \pm 0.0211$ | 1.1564 | 1.0618 | Robust Operating Plateau |
| **0.12** | **$2.00 \pm 0.00$** | **$68.09 \pm 7.31$** | **\$2.780 \pm 0.136** | **$0.1923 \pm 0.0211$** | **1.1564** | **1.0607 (Knee Point)** | **Robust Operating Plateau** |
| **0.15** | $2.00 \pm 0.00$ | $70.17 \pm 5.78$ | \$2.742 \pm 0.116 | $0.1912 \pm 0.0226$ | 1.1666 | 1.0699 | Robust Operating Plateau |
| **0.18** | $1.80 \pm 0.40$ | $69.47 \pm 5.59$ | \$2.744 \pm 0.110 | $0.2238 \pm 0.0299$ | 1.1609 | 1.0628 | Uncorrected Drift Region |
| **0.22** | $1.00 \pm 0.00$ | $73.26 \pm 4.82$ | \$2.687 \pm 0.056 | $0.2596 \pm 0.0394$ | 1.1820 | 1.0738 | Uncorrected Drift Region |
| **0.35** | $0.80 \pm 0.40$ | $74.72 \pm 2.30$ | \$2.739 \pm 0.213 | $0.3584 \pm 0.0157$ | 1.2054 | 1.0929 | Uncorrected Drift Region |
| **0.50** | $0.00 \pm 0.00$ | $80.15 \pm 4.31$ | \$3.100 \pm 0.067 | $0.3947 \pm 0.0352$ | 1.3259 | 1.1933 | Uncorrected Drift Region |
---

### 7.4 Physical Hardware Verification & Signal Processing
The physical mechatronic breadboard testbed, evaluated in **High-Fidelity Electro-Mechanical Emulator (with Unmodeled Material & Thermal Disturbances)** mode, verifies real-time sensor processing, decoupling observation from data generation:
- **Dual-Sensor Wear Estimation (INA226 + ADXL345 Fusion):** Under unmodeled material hardness fluctuations (AR(1) $\delta_{\text{mat}} \in [-4\%, +4\%]$), calibration mismatch, and intermittent chatter bursts ($VB > 65\,\mu\text{m}$), the EKF achieves $\text{RMSE} = 1.087\,\mu\text{m}$, $\text{MAE} = 0.873\,\mu\text{m}$, and $R^2 = 0.9984$ (improving over single-sensor vibration by $50.75\%$ and single-sensor power by $10.35\%$). In the comprehensive 60-second operational profile, full-trajectory tracking achieves $\text{RMSE} = 1.763\,\mu\text{m}$ and $R^2 = 0.9951$.
- **Impulse Rejection:** 1D Kalman filter combined with 3-tap median pre-filtering rejects spurious PWM motor spikes ($+25\,\text{W}$) within $5\,\text{ms}$.
- **Vibration RMS:** 20-sample sliding window ($100\,\text{ms}$) maintains baseline acceleration tracking during steady-state cutting ($0.85\,\text{g}$ to $1.96\,\text{g}$ worn chatter).

![Hardware Verification Telemetry](tests/phase3_hardware_verification.png)

---

### 7.5 Inter-Machine Transport & Setup Latency Sensitivity ($\tau_{\text{trans}}$)

To address shop-floor material handling dynamics (AGV transfer / pallet shuttle), inter-machine transport latency $\tau_{\text{trans}} \in [0.0, 1.0, 2.0, 3.0]\,\text{min}$ was evaluated across 15 test seeds (`TEST_SEEDS = [301..315]`) under Regime B stochastic wear drift on both **Mk01** ($10 \times 6$, 55 operations) and **Mk03** ($15 \times 8$, 150 operations). All 4 dispatching and rescheduling paradigms were evaluated ($2 \times 4 \times 4 = 32$ audited configurations, 480 simulated shop-floor runs):

#### (a) Benchmark Mk01 (10 jobs $\times$ 6 machines, 55 operations)

| $\tau_{\text{trans}}$ (min) | Method | Makespan $C_{\max}$ (min) | TOU Cost (\$) | TWT (min) | Tool Breakdowns |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0.0**<br>*(Baseline)* | NSGA-III (Open-Loop) | $83.64 \pm 11.90$ | \$2.570 \pm 0.233 | $362.67 \pm 101.70$ | **0.87** |
|  | Right-Shift Rescheduling | $79.58 \pm 5.95$ | \$2.594 \pm 0.152 | $321.09 \pm 95.12$ | 0.00 |
|  | Standalone Edge PPO | $110.42 \pm 3.77$ | \$3.245 \pm 0.136 | $673.57 \pm 45.85$ | 0.00 |
|  | **Proposed Cloud-Edge DT** | **$86.00 \pm 9.79$** | **\$2.793 \pm 0.183** | **$329.99 \pm 60.47$** | **0.00** |
| **1.0** | NSGA-III (Open-Loop) | $93.70 \pm 11.95$ | \$2.549 \pm 0.194 | $457.73 \pm 113.19$ | **0.87** |
|  | Right-Shift Rescheduling | $91.46 \pm 7.28$ | \$2.575 \pm 0.123 | $438.39 \pm 119.87$ | 0.00 |
|  | Standalone Edge PPO | $127.27 \pm 4.58$ | \$3.184 \pm 0.134 | $878.34 \pm 56.09$ | 0.00 |
|  | **Proposed Cloud-Edge DT** | **$98.12 \pm 12.15$** | **\$2.830 \pm 0.218** | **$436.99 \pm 64.78$** | **0.00** |
| **2.0** | NSGA-III (Open-Loop) | $104.69 \pm 12.39$ | \$2.566 \pm 0.182 | $570.67 \pm 126.71$ | **0.87** |
|  | Right-Shift Rescheduling | $104.12 \pm 8.79$ | \$2.522 \pm 0.111 | $571.72 \pm 142.02$ | 0.00 |
|  | Standalone Edge PPO | $143.61 \pm 5.12$ | \$3.107 \pm 0.104 | $1070.74 \pm 55.69$ | 0.00 |
|  | **Proposed Cloud-Edge DT** | **$117.77 \pm 14.31$** | **\$2.813 \pm 0.172** | **$606.41 \pm 147.90$** | **0.00** |
| **3.0** | NSGA-III (Open-Loop) | $116.87 \pm 13.58$ | \$2.542 \pm 0.165 | $697.83 \pm 140.95$ | **0.87** |
|  | Right-Shift Rescheduling | $117.43 \pm 10.21$ | \$2.480 \pm 0.133 | $716.39 \pm 160.20$ | 0.00 |
|  | Standalone Edge PPO | $159.54 \pm 5.55$ | \$3.196 \pm 0.102 | $1265.61 \pm 60.59$ | 0.00 |
|  | **Proposed Cloud-Edge DT** | **$130.24 \pm 18.73$** | **\$2.937 \pm 0.162** | **$764.10 \pm 135.16$** | **0.00** |

#### (b) Benchmark Mk03 (15 jobs $\times$ 8 machines, 150 operations)

| $\tau_{\text{trans}}$ (min) | Method | Makespan $C_{\max}$ (min) | TOU Cost (\$) | TWT (min) | Tool Breakdowns |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0.0**<br>*(Baseline)* | NSGA-III (Open-Loop) | $514.37 \pm 25.12$ | \$14.280 \pm 0.517 | $5239.80 \pm 815.41$ | **9.67** |
|  | Right-Shift Rescheduling | $420.24 \pm 26.54$ | \$13.479 \pm 0.435 | $3605.98 \pm 529.67$ | **0.07** |
|  | Standalone Edge PPO | $438.62 \pm 44.32$ | \$26.469 \pm 1.338 | $5093.45 \pm 591.89$ | **6.27** |
|  | **Proposed Cloud-Edge DT** | **$462.69 \pm 45.99$** | **\$13.883 \pm 0.677** | **$4621.78 \pm 889.63$** | **0.00** |
| **1.0** | NSGA-III (Open-Loop) | $550.77 \pm 28.69$ | \$14.257 \pm 0.638 | $5839.26 \pm 831.67$ | **9.67** |
|  | Right-Shift Rescheduling | $458.31 \pm 32.21$ | \$13.933 \pm 0.468 | $4241.42 \pm 579.11$ | **0.07** |
|  | Standalone Edge PPO | $466.56 \pm 49.07$ | \$26.117 \pm 1.703 | $5443.43 \pm 770.64$ | **6.73** |
|  | **Proposed Cloud-Edge DT** | **$474.54 \pm 28.76$** | **\$14.189 \pm 0.776** | **$4943.58 \pm 698.14$** | **0.00** |
| **2.0** | NSGA-III (Open-Loop) | $589.88 \pm 32.56$ | \$14.257 \pm 0.750 | $6498.95 \pm 850.14$ | **9.67** |
|  | Right-Shift Rescheduling | $498.53 \pm 36.59$ | \$14.018 \pm 0.342 | $4919.05 \pm 649.70$ | **0.07** |
|  | Standalone Edge PPO | $473.42 \pm 44.68$ | \$25.390 \pm 1.185 | $5035.02 \pm 661.90$ | **6.20** |
|  | **Proposed Cloud-Edge DT** | **$507.72 \pm 39.07$** | **\$14.373 \pm 0.710** | **$5615.45 \pm 992.88$** | **0.00** |
| **3.0** | NSGA-III (Open-Loop) | $630.07 \pm 37.35$ | \$14.214 \pm 0.849 | $7185.78 \pm 891.60$ | **9.67** |
|  | Right-Shift Rescheduling | $539.83 \pm 41.43$ | \$14.017 \pm 0.390 | $5628.34 \pm 725.94$ | **0.07** |
|  | Standalone Edge PPO | $483.63 \pm 34.62$ | \$25.007 \pm 1.036 | $5152.95 \pm 700.09$ | **5.60** |
|  | **Proposed Cloud-Edge DT** | **$525.40 \pm 43.53$** | **\$14.688 \pm 0.859** | **$5760.48 \pm 810.57$** | **0.00** |

![Transport Latency Sensitivity](tests/phase5_transport_sensitivity.png)

#### Key Physical & Methodological Insights:
1. **Boundary Parity ($\tau_{\text{trans}} = 0.0\,\text{min}$):** At zero transport delay, all 4 methods reproduce the exact multi-instance figures in `phase5_wilcoxon_benchmark_table.csv` ($86.00 \pm 9.79\,\text{min}$, $\$2.793 \pm 0.183$, $329.99 \pm 60.47\,\text{min}$, $0.00$ breakdowns for Proposed DT on Mk01; $462.69 \pm 45.99\,\text{min}$, $\$13.883 \pm 0.677$, $4621.78 \pm 889.63\,\text{min}$, $0.00$ breakdowns on Mk03) with zero deviation ($< 10^{-6}$).
2. **Monotonic Makespan Scaling:** For all schedule-following methods (`Proposed Cloud-Edge DT`, `Right-Shift Rescheduling`, `NSGA-III`), makespan scales strictly monotonically ($+12.1 \dots +14.7\,\text{min}$ per $1.0\,\text{min}$ transport latency on Mk01; $+20.8 \dots +31.8\,\text{min}$ per $1.0\,\text{min}$ latency on Mk03) as inter-machine transfers directly lengthen precedence paths.
3. **Zero-Breakdown Robustness:** `Proposed Cloud-Edge DT` preserves **$0.00$ tool breakdowns** across all $\tau_{\text{trans}} \in [0.0, 3.0]\,\text{min}$ on both small and large shop scales, in stark contrast to $0.87$ failures on Mk01 and $9.67$ failures on Mk03 suffered by open-loop baselines.
4. **Physical Mechanism for Non-Monotonic TOU Electricity Cost:**
   Unlike makespan, total TOU electricity cost exhibits subtle non-monotonic oscillations across transport delays (e.g., Mk01 Proposed DT cost is $\$2.793$ at $\tau=0$, $\$2.830$ at $\tau=1$, drops to $\$2.813$ at $\tau=2$, and rises to $\$2.937$ at $\tau=3$). This is physically governed by **phase shifts across the 100-minute cyclic TOU tariff**:
   - Off-Peak: $[0, 25)\,\text{min} \pmod{100}$ at $\$0.06/\text{kWh}$
   - On-Peak: $[25, 60)\,\text{min} \pmod{100}$ at $\$0.20/\text{kWh}$
   - Mid-Peak: $[60, 100)\,\text{min} \pmod{100}$ at $\$0.12/\text{kWh}$

   Injecting transport delays delays the start times of subsequent heavy machining operations by non-integer multiples of tariff phases. When an energy-intensive cutting block that previously occurred during an On-Peak window ($25 \dots 60\,\text{min}$ at $\$0.20/\text{kWh}$) is shifted by transport latency into an Off-Peak ($0 \dots 25\,\text{min}$ at $\$0.06/\text{kWh}$) or Mid-Peak ($60 \dots 100\,\text{min}$ at $\$0.12/\text{kWh}$) window, the substantially lower unit rate ($\Delta C = -\$0.14/\text{kWh}$ or $-\$0.08/\text{kWh}$) offsets the marginal standby/idle energy consumed during transport transit, creating a localized decrease in total electricity expenditure despite an expanded makespan.

---

### 7.6 Unified Table of Experimental Regimes

To ensure complete experimental reproducibility, the research protocol distinguishes three formal regimes:

| Regime | Definition & Dynamics | Uncertainty Parameters | Objective Scope | Seed Set | Protocol Phase |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Regime A** | Nominal Deterministic | No wear acceleration ($\eta_{\text{wear}} = 1.0$), nominal processing times, static availability | Upper-bound reference frontiers & Cloud NSGA-III baseline generation | Nominal seed (42) | Phase 1 & Cloud Pareto Front |
| **Regime B** | Stochastic Continuous Wear Drift | Real-world $+15\% \dots +35\%$ stochastic flank wear drift, speed-dependent power dynamics | Multi-instance generalization, non-parametric Wilcoxon tests, and architectural ablation | 15 `TEST_SEEDS` (`[301..315]`) | Phase 2, Phase 5 Multi-Scale & Ablation |
| **Regime C** | Stochastic Dual Macro-Shock HIL | Continuous wear drift plus 2 macro-disturbances (Op 18..22 hardness surge $+45\%..+75\%$; Op 36..40 emergency stoppage) | Hardware-in-the-Loop closed-loop validation and CDI threshold sensitivity | 10 `VAL_SEEDS` (`[201..210]`) | Phase 4 HIL & Phase 5 CDI Sensitivity |

---

### 7.7 Zero-Shot Scale Generalization Protocol

A key practical barrier to DRL in manufacturing is retraining cost. The proposed Edge `MaskablePPO` policy achieves **Zero-Shot Scale Generalization** through three scale-invariant architectural principles:

1. **Backlog Distribution Quantiles:** The observation space extracts 5 statistical percentiles ($10\%, 25\%, 50\%, 75\%, 90\%$) of remaining workload, processing times, and urgency slack across shop queues, decoupling state dimension from the number of active jobs.
2. **Normalized Machine Vectors:** Each workstation state is normalized against its own nominal envelope: current flank wear ratio $w_k(t) / w_{\text{max}}$, dynamic power drift increment $\Delta P_{\text{cut}} / P_{\text{rated}}$, and normalized time until release.
3. **Cloud Prior Alignment:** Cloud guidance is structured as relative direction vectors (recommended cutting speed level and assignment compatibility) rather than rigid absolute timestamps.

**Training & Transfer:** The policy is trained strictly on `Mk01` ($10 \times 6$, 55 operations) using `TRAIN_SEEDS = [42, 101, 2026]` over 66,000 steps, exported to ONNX Runtime (`edge_policy.onnx`), and deployed **Zero-Shot** to `Mk02` (58 ops), `Mk03` (150 ops), `Mk04` (90 ops), and `Mk05` (106 ops) on `TEST_SEEDS = [301..315]`. As demonstrated in Table 7.1, it successfully maintains **0.00 tool breakdowns** across all scales without requiring any retraining or fine-tuning.

---

### 7.8 Academic Literature Positioning Table

Comparison of the proposed collaborative framework against recent state-of-the-art literature in green FJSP, tool degradation dynamics, and industrial digital twins:

| Study | Primary Methodology | Dynamic Disturbances Considered | Multi-Objective Scope | Tool Wear & PM Co-design | Speed-Scaling (Green FJSP) | Cloud-Edge Collaborative DT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Wu et al. (2022)** | Deep Reinforcement Learning (PPO) | Machine breakdowns | Makespan, Energy | Heuristic / Periodic | Binary (Processing / Idle) | No (Centralized DRL) |
| **Tian et al. (2023)** | Multi-objective Evolutionary (MOEA) | Static / Deterministic | Makespan, Cost, Carbon | Offline wear model | Fixed cutting speeds | No (Offline Cloud Solver) |
| **Mohammad-Rafiei et al. (2024)** | Metaheuristics (NSGA-II) | Stochastic wear drift | Makespan, Energy | Offline wear degradation | Parameterized speed levels | No (Offline / Open-Loop) |
| **Wang et al. (2023)** | Cloud-Edge Digital Twin | Job cancellations | Makespan, Machine load | None (Zero wear modeling) | Constant cutting power | Asynchronous Cloud-Edge |
| **Zhang et al. (2022)** | Edge-Cloud DRL Framework | Random rush orders | Makespan, Tardiness | None (Static power profile) | Fixed cutting speed | Hierarchical Cloud-Edge |
| **Gao et al. (2024)** | Digital Twin 3D Visualization | Random delays | Makespan | Monitored (Visual only) | None (Static machine states) | Partial (Cloud-centric monitoring) |
| **Zhao & Xu (2025)** | Cloud-Edge Collaborative Framework | Arrival of urgent jobs | Makespan, Tardiness | None | Discrete speed levels | Periodic Rescheduling (No CDI) |
| **Zhu et al. (2024)** / **Zhu (2026)** | Hierarchical Cloud-Edge Optimization | Stochastic process duration | Makespan, Energy | Breakdown probability | Heuristic speed selection | Edge dispatching only |
| **Gong et al. (2018)** | Hybrid Genetic Algorithm | Static / Deterministic | Makespan, TOU Cost | None | Static speed selection | No (Single-tier offline) |
| **Zhao et al. (2025)** | Multi-Objective Evolutionary Algorithm | Dynamic job arrivals | Makespan, TOU, Tardiness | None | Offline speed scheduling | No (Centralized Cloud) |
| **Burmeister (2024)** | Machine State Control (Standby) | Buffer starvation | Energy consumption | None | Binary On/Off states | Local PLC logic |
| **Proposed Framework** | **Closed-Loop Co-Design: Cloud NSGA-III + Edge MaskablePPO + Dual-Sensor Fusion + Shift-Normalized CDI** | **Continuous tool wear drift ($+15\%..+35\%$) + Stochastic dual macro-shocks (Hard alloy & Stoppage)** | **Makespan ($C_{\max}$), TOU Electricity Cost, Total Weighted Tardiness (TWT)** | **Physics-informed Taylor dynamic wear + Dual-sensor telemetry (INA226 + ADXL345) + Pre-operation predictive PM ($0.00$ breakdowns)** | **Sub-millisecond real-time speed adaptation ($s \in \{1, 2, 3\}$) synchronized with Cyclic TOU Tariff & Peak Limit ($P_{\text{limit}} = 22.0\,\text{kW}$)** | **Full Co-Design: Event-triggered re-sync ($\theta_{\text{th}} = 0.12$) reducing Cloud calls by 80% while retaining sub-millisecond edge reactivity (0.0132 ms forward, 0.1506 ms step)** |

*Academic Positioning & Novelty Formulation:* In strict alignment with rigorous scholarly standards, we do not claim to be the "first" to utilize Digital Twins, model tool wear, or apply MaskablePPO in manufacturing scheduling. Rather, our scientific contribution is situated in direct contrast to four distinct clusters of pioneering literature:
1. **Contrasted with Tian et al. (2023, Mathematics) and Mohammad-Rafiei et al. (2024):** While they modeled cutting tool degradation and cutting speed-dependent power drift, their solutions relied on static, offline evolutionary algorithms. They lacked real-time, closed-loop speed adaptation to safeguard tools against random stochastic wear spikes and did not evaluate dynamic cyclic TOU electricity tariffs.
2. **Contrasted with Gao et al. (2024, RCIM), Wang et al. (2023, RCIM), and Zhang et al. (2022, IEEE TII):** Although they established distributed Cloud-Edge Digital Twin and DRL frameworks for dynamic scheduling, they treated machining power consumption as idealized static constants, neglecting the physical tool flank wear dynamics that induce significant active power drift ($+10.98\%$) and catastrophic in-cut tool breakage.
3. **Contrasted with Wu et al. (2022) and Zhu et al. (2024) / Zhu (2026):** We substantially generalize the concept of recessive disturbance accumulation and incident coefficients into a shift-normalized dual-deviation Cumulative Disturbance Index $\text{CDI}(t)$, rigorously identifying a robust operating plateau ($\theta_{\text{th}} \in [0.08, 0.15]$) that minimizes total system operational cost.
4. **Contrasted with Zhao et al. (2025, ASOC), Gong et al. (2018), and Burmeister (2024):** We extend the classic MOFJSP-TOU problem from static offline scheduling to mechatronic, hardware-in-the-loop 3-state power control (Opto-isolated SSR zero-loss standby) integrated with factory-level peak demand capping ($P_{\text{limit}} = 22.0\,\text{kW}$).

---

### 7.9 Balanced Scientific Narrative & Physical Boundary Limits

To maintain objective academic rigor and transparent reproducibility, we explicitly highlight the competitive baseline dynamics, trade-offs, operational boundaries, and physical testbed constraints:

1. **Honest Assessment of `Right-Shift Rescheduling`:**
   - *A Highly Competitive Reactive Baseline:* When inheriting high-quality global Pareto baseline schedules from Cloud NSGA-III, simple Right-Shift rescheduling serves as an exceptionally formidable reactive baseline. On small problem instance `Mk01`, no statistically significant difference in Hypervolume was detected between Right-Shift and the Proposed DT ($p_{\text{holm}} = 0.0747$). On `Mk02` ($p_{\text{holm}} = 0.0015$), `Mk03` ($p_{\text{holm}} = 0.0009$), `Mk04` ($p_{\text{holm}} = 0.0214$), and `Mk05` ($p_{\text{holm}} = 0.0009$), Right-Shift secures higher Hypervolume by strictly preserving the global assignment topology. Under stochastic wear drift, Right-Shift incurs minor tool breakdowns ($0.07$ on Mk03 and Mk05).
   - *The Decisive Advantage of Proposed DT:* The true operational superiority of the Proposed DT over Right-Shift stems from active cutting speed-scaling ($s \in \{1, 2, 3\}$), dynamic tool wear tracking via EKF, and CDI-triggered Cloud re-synchronization. Proposed DT completely eliminates tool breakdowns ($0.00$ failures across all 5 benchmark instances), achieves the benchmark lowest TOU electricity cost on bottleneck `Mk05` ($\$11.198 \pm 0.539$ vs $\$12.343 \pm 0.783$ for Right-Shift, saving $9.28\%$), and provides hard real-time speed adjustments to prevent peak-tariff penalties.

2. **Objective Analysis of `Standalone Edge PPO (No Cloud Prior)`:**
   - *Local Reactivity on Dense Benchmarks:* On dense instances with tight deadlines (`Mk03–Mk05`), Standalone Edge PPO operates on pure local urgency slack reflex, occasionally cutting makespan (e.g. $344.81 \pm 25.02\,\text{min}$ vs $408.76 \pm 29.48\,\text{min}$ on Mk05; $151.79\,\text{min}$ vs $161.25\,\text{min}$ on Mk04) or TWT ($4896.0\,\text{min}$ vs $5395.5\,\text{min}$ on Mk05).
   - *The Penalty of Missing Global Guidance:* Lacking the global combinatorial perspective of Cloud NSGA-III (No Cloud Prior), Standalone Edge PPO accelerates machining speeds indiscriminately through expensive On-Peak tariff windows ($[25..60]\,\text{min}$ at $\$0.20/\text{kWh}$) and creates localized machine bottlenecks. On Mk01, makespan degrades by $+28.4\%$ ($110.42\,\text{min}$ vs $86.00\,\text{min}$) and TWT doubles to $673.6\,\text{min}$ vs $330.0\,\text{min}$ ($+104.1\%$). Across all instances, it suffers frequent tool breakdowns ($0.80 - 6.27$ failures per episode) and inflates TOU electricity costs by **$+16.2\%$ to $+90.7\%$** ($+16.2\%$ on Mk01, $+90.7\%$ on Mk03 from $\$13.883$ to $\$26.469$, $+47.6\%$ on Mk04 from $\$4.632$ to $\$6.835$, $+38.0\%$ on Mk05 from $\$11.198$ to $\$15.457$) compared to the Proposed DT.

3. **Discrepancy Between Hypervolume (HV) and IGD on Benchmark `Mk05`:**
   - *The Bottleneck Machine Dilemma:* `Mk05` poses an extreme structural bottleneck ($15$ jobs comprising $106$ operations allocated across only $4$ machines, averaging $26.5$ operations per machine).
   - *Trade-off Mechanism:* In this constrained topology, the Proposed DT prioritizes absolute physical tool preservation ($0.00$ tool breakdowns vs $9.20$ for NSGA-III, $9.87$ for MOEA-D, and $12.67$ for RVEA) and aggressive TOU cost minimization (achieving the lowest electricity bill across all methods: $\$11.198 \pm 0.539$ vs $\$12.343 \pm 0.783$ for Right-Shift, saving $9.28\%$). Decelerating to $s=1$ during peak tariff hours to avoid demand penalties unavoidably extends completion times for queued jobs, resulting in higher TWT ($5395.5 \pm 800.5\,\text{min}$) and a compressed Hypervolume along the tardiness axis.
   - *Pareto Convergence:* Crucially, while boundary solutions are truncated, the non-dominated solutions generated by Proposed DT converge exceptionally close to the true reference front center, achieving $\text{IGD} = 0.3619 \pm 0.0715$, outperforming open-loop metaheuristics (NSGA-III: $0.4873 \pm 0.1248$, MOEA-D: $0.6129 \pm 0.1798$, RVEA: $0.5436 \pm 0.0893$).

4. **Small-Scale vs. Large-Scale Trade-offs:**
   - *Small Problem Regimes (Mk01, Mk02):* Open-loop evolutionary algorithms (`MOEA-D`, `NSGA-III`, `NSGA-II`, `RVEA`) omit dynamic condition-based tool replacements, suffering $0.33 - 0.87$ tool breakdowns per run even on small jobs.
   - *Large Industrial Regimes (Mk03–Mk05):* In environments with long machining horizons and bottleneck machines, open-loop methods collapse under cumulative tool wear, suffering up to $9.67$ breakdowns on Mk03 and $12.67$ breakdowns on Mk05. In contrast, Proposed DT eliminates all breakdowns ($0.00$ failures across all 5 benchmark instances), achieving statistically significant Hypervolume superiority over all 4 open-loop metaheuristics on Mk05 ($p_{\text{holm}} \le 0.0011$) and beating NSGA-III ($p_{\text{holm}} = 0.0214$) and NSGA-II ($p_{\text{holm}} = 0.0410$) on Mk03.

5. **Scale-Dependent Hazard of Variant 1 (w/o Wear Feedback):**
   - On `Mk01` ($55$ operations): Variant 1 experiences $0.80$ breakdowns and an inflated TWT of $373.68 \pm 129.18\,\text{min}$ ($+13.24\%$ vs $329.99\,\text{min}$ for Full Proposed). Sensor-blind execution delays preventive maintenance, demonstrating the acute danger of *Blind Closed-Loop Control*.
   - On `Mk03` ($150$ operations): Under heavy cumulative wear, Variant 1 suffers $7.60$ breakdowns per run (vs. $0.00$ for Full Proposed), driving makespan to $511.54 \pm 29.06\,\text{min}$ ($+10.56\%$) and TWT to $5226.80 \pm 801.80\,\text{min}$ ($+13.09\%$).

6. **The Nature of Variant 2 as an Operational Trade-Off:**
   - Locking cutting speed to nominal $s=2$ is an operational design choice rather than an unmitigated defect: it yields modest reductions in makespan ($-7.75\%$ on Mk01 from $86.00$ to $79.34\,\text{min}$, $-10.60\%$ on Mk03 from $462.69$ to $413.64\,\text{min}$) and TWT ($-10.89\%$ on Mk01, $-18.85\%$ on Mk03) because machines never drop to conservative speed $s=1$.
   - However, it loses the agile capability to throttle cutting during high-tariff windows ($[25..60]\,\text{min}$ at $\$0.20/\text{kWh}$), increasing TOU electricity costs by $+8.53\%$ on Mk01 (from $\$2.793$ to $\$3.031$) and $+1.80\%$ on Mk03 (from $\$13.883$ to $\$14.133$).

7. **Physical Proof-of-Concept Hardware Limits:**
   - The physical testbed operates as a hardware-in-the-loop laboratory proof-of-concept using an ESP32 embedded node, INA226 power monitor, ADXL345 accelerometer, and a 24V DC CNC spindle proxy.
   - Real-world deployment on heavy 380V 3-phase industrial machine tools will introduce additional challenges, including coolant ingress, high-frequency electromagnetic noise from variable frequency drives (VFDs), and complex modal vibrations requiring industrial-grade enclosure and fiber-optic communication channels.

---

## 8. Citation

If you find this work, codebase, or experimental artifacts useful in your research, please cite our paper:

```bibtex
@article{nguyen2026cloudedgefjsp,
  title     = {Cloud-Edge Collaborative Digital Twin for Dynamic Green Flexible Job Shop Scheduling with Speed-Scaling and Tool Wear Resilience},
  author    = {Nguyen, Minh-Nguyen and Collaborators},
  journal   = {IEEE Transactions on Industrial Informatics},
  year      = {2026},
  volume    = {PP},
  number    = {99},
  pages     = {1--12},
  doi       = {10.1109/TII.2026.XXXXXXX}
}
```

---

## 9. License

This project is licensed under the terms of the [MIT License](LICENSE). All benchmark data, trained neural network weights, firmware, and test suites are freely available for academic and industrial research.

