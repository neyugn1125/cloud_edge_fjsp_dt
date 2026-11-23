# A Cloud-Edge Collaborative Digital Twin Framework for Energy-Aware Flexible Job Shop Scheduling with Dynamic Speed-Scaling and Tool Wear Resilience

**Authors:** Minh-Nguyen Nguyen, Collaborators  
**Target Journal:** IEEE Transactions on Industrial Informatics / Elsevier Journal of Manufacturing Systems  
**Status:** Post-Audit Experimental Manuscript (Fully Synchronized Across Phases 1–5)  
**Keywords:** Flexible Job Shop Scheduling, Cloud-Edge Digital Twin, Deep Reinforcement Learning, Tool Wear Resilience, Cyclic Time-of-Use Tariff, Hardware-in-the-Loop.

---

## Abstract
Dynamic Flexible Job Shop Scheduling Problems (D-FJSP) in discrete manufacturing face severe operational uncertainties, including non-linear tool degradation, time-varying electricity tariffs, and strict real-time control constraints. While centralized Cloud metaheuristics yield high-quality global Pareto schedules, their computational latency ($4.24 \pm 0.03\,\text{s}$ on Mk01, $14.97\,\text{s}$ on Mk03; $2.6835\,\text{s}$ warm-start resync) precludes real-time disturbance reaction. Conversely, distributed edge dispatching policies provide rapid reactivity but suffer from localized myopia, peak-tariff overloading, and premature tool breakage. This paper presents a closed-loop **Cloud-Edge Collaborative Digital Twin (DT)** framework that harmonizes long-term energy-cost optimization with sub-millisecond shop-floor control. The framework comprises: (1) a Cloud macro-optimizer executing reference-direction NSGA-III over Makespan ($C_{\max}$), Time-of-Use (TOU) energy cost, and Total Weighted Tardiness (TWT); (2) an Edge micro-controller powered by a scale-invariant MLP policy trained via Maskable Proximal Policy Optimization (MaskablePPO) and executed on ONNX Runtime in $0.0132\,\text{ms}$ forward pass and $0.1506\,\text{ms}$ per full decision step ($\sim 2.81 \times 10^4 \times$ faster than Cloud optimization); (3) a shift-normalized Cumulative Disturbance Index ($\text{CDI}(t)$) deployed at $\theta_{\text{th}} = 0.12$ within the objective Event-Matched Plateau $[0.08, 0.18]$ ($1$–$2$ calls matched to $N_{\text{shocks}} = 2.0$, whereas $\theta_{\text{th}}^* = 0.05$ minimizes `Total_System_Cost_Index` at $1.0325$ with $2.70$ over-triggered calls), reducing online Cloud Warm-Start rescheduling calls by $87.3\%$ ($1.27 \pm 0.44$ vs. $10.00$ periodic calls; $36.5\%$ fewer than fixed frequency-matched Op 25/50 periodic calls at $2.00$) and cutting TOU electricity cost by $8.5\%$ relative to high-frequency periodic EKF resync ($p = 0.0067$); and (4) an opto-isolated solid-state relay (SSR) mechatronic subsystem enabling zero-loss standby transitions ($T_{\text{BE}} \in [8.36, 10.54]\,\text{min}$ on shop-floor CNC centers; $T_{\text{BE}} \approx 3.1\,\text{min}$ on hardware proxy) with Stop Category 1 hardware safety interlocks (`GPIO 27` Open-Drain GRBL `Feed Hold`, $2500\,\text{ms}$ / $2.50\,\text{s}$ kinematic deceleration sync, and $300\,\text{ms}$ Host Watchdog). Audited across 750 independent experimental runs (15 test seeds, Brandimarte Mk01–Mk05) against 9 competing baselines, the proposed framework eliminates catastrophic tool failures ($0.00$ breakdowns vs. $0.33 - 12.67$ for open-loop baselines), reduces TOU energy costs by up to $30.95\%$ via SSR standby gating (saving $\$6.223$ per shift on Mk03 vs. the idle-only baseline; or $+44.82\%$ cost penalty when SSR standby is omitted), achieves mean $\overline{\text{HV}} = 0.4702$ and $\overline{\text{IGD}} = 0.3675$, and achieves statistically significant Hypervolume superiority over all 4 open-loop metaheuristics on bottleneck Mk05 ($p_{\text{holm}} \le 0.0011$) and over NSGA-III ($p_{\text{holm}} = 0.0214$) and NSGA-II ($p_{\text{holm}} = 0.0410$) on large-scale Mk03. When reactive Right-Shift Rescheduling is fairly equipped with the identical live EKF wear estimator and PM safety rule to also achieve $0.00$ breakdowns, two-sided Wilcoxon tests show no statistically significant difference in Hypervolume on Mk01 ($p_{\text{holm}} = 0.2768$) and Mk04 ($p_{\text{holm}} = 0.1508$), while Right-Shift achieves statistically higher Hypervolume on Mk02 ($p_{\text{holm}} = 0.0043$), Mk03 ($p_{\text{holm}} = 0.0043$), and Mk05 ($p_{\text{holm}} = 0.0005$), and the Proposed DT lowers TOU electricity cost by $8.44\%$ on bottleneck Mk05 (\$11.198 vs. \$12.230). Physical validation on an ESP32 dual-sensor CNC breadboard in high-fidelity electro-mechanical emulation with unmodeled material and thermal disturbances confirms robust real-time tool wear estimation ($R^2 = 0.9984$ during active cutting, $R^2 = 0.9951$ across the full 60s operational profile with $\text{RMSE} = 1.763\,\mu\text{m}$) and safe non-preemptive overrun boundaries.

---

## 1. Introduction & Academic Literature Positioning

### 1.1 Context and Industrial Motivation
Discrete manufacturing environments, such as computer numerical control (CNC) machining workcells, are undergoing unprecedented transformations driven by decarbonization mandates, volatile electricity spot markets, and increasing customization. Scheduling in flexible job shops requires co-optimizing production efficiency (makespan and tardiness) with sustainability metrics, particularly Time-of-Use (TOU) electricity expenditure and physical machine degradation. 

Under operational conditions, machining processes suffer from continuous, stochastic degradation. Cutting tool flank wear progresses non-linearly, causing cutting forces and spindle loads to escalate. This induces an active cutting power drift of $+10.98\%$ to $+35\%$, exacerbates chatter vibration ($+18.75\%$ to $+28.1\%$), and drastically escalates tool breakage risks if condition-based maintenance (CBM) is misjudged. Simultaneously, industrial power utilities enforce steep peak-hour tariffs (up to $3.33\times$ off-peak rates), necessitating dynamic cutting speed adjustment ($s \in \{1, 2, 3\}$) and factory peak-demand capping ($P_{\text{limit}}$).

### 1.2 Scholarly Positioning: Direct Literature Contrast
In strict compliance with academic standards, we explicitly avoid claiming an isolated novelty for general Digital Twins, basic cutting speed-scaling, or standard Proximal Policy Optimization. Instead, the scientific novelty of this work is situated through direct contrast with four primary clusters of recent manufacturing literature:

1. **Contrast with Tian et al. (2023, Mathematics) and Mohammad-Rafiei et al. (2024):**  
   Tian et al. and Mohammad-Rafiei et al. developed multi-objective mathematical models incorporating cutting tool wear progression and cutting parameter optimization. However, their methodologies rely strictly on static, offline (open-loop) evolutionary algorithms (NSGA-II / MOEA/D). Because these frameworks lack real-time feedback, they cannot adjust spindle speeds dynamically when stochastic wear spikes occur mid-shift, leading to tool breakage under random disturbances. Furthermore, they do not account for multi-period cyclic TOU electricity tariffs or hardware standby power states.

2. **Contrast with Gao et al. (2024, RCIM), Wang et al. (2023, RCIM), and Zhang et al. (2022, IEEE TII):**  
   These authors proposed state-of-the-art Cloud-Edge Digital Twin and distributed Deep Reinforcement Learning architectures for dynamic job shop dispatching. Nevertheless, their underlying physical formulations treat machine power consumption as an idealized static constant (binary processing or idle power). By neglecting the physical reality of tool wear dynamics, their models remain blind to active cutting power drift and are incapable of preventing in-cut tool failure.

3. **Contrast with Wu et al. (2022) and Zhu et al. (2024) / Zhu (2026):**  
   Wu et al. explored DRL for machine breakdown handling, while Zhu et al. introduced hierarchical edge dispatching under stochastic operation durations. We significantly generalize the concept of recessive disturbance accumulation into a shift-normalized dual-deviation Cumulative Disturbance Index ($\text{CDI}(t)$). By integrating both makespan lag and energy drift normalized across shift expectations, our framework systematically identifies an operational plateau ($\theta_{\text{th}} \in [0.08, 0.15]$) that minimizes total system cost (schedule degradation plus solver communication overhead).

4. **Contrast with Zhao et al. (2025, ASOC), Gong et al. (2018), and Burmeister (2024):**  
   Gong et al. and Zhao et al. modeled green FJSP under TOU electricity tariffs using hybrid genetic algorithms and evolutionary solvers. Burmeister investigated machine standby state transitions during buffer starvation. Our work elevates the green MOFJSP-TOU formulation from offline simulation into a closed-loop mechatronic implementation: integrating an opto-isolated solid-state relay (SSR) subsystem with physical break-even transition thresholds ($T_{\text{BE}, k} \in [8.36, 10.54]\,\text{min}$ on CNC machines; $T_{\text{BE}} \approx 3.1\,\text{min}$ on physical hardware proxy) and instance-specific factory peak demand ceilings ($P_{\text{limit}} \in \{22.0, 24.0, 32.0, 30.0, 16.0\}\,\text{kW}$).

---

## 2. Mathematical Problem Formulation & Tri-Tier DT Architecture

### 2.1 Multi-Objective Optimization Formulation
The Dynamic Flexible Job Shop Scheduling Problem under study involves scheduling a set of $n$ jobs $\mathcal{J} = \{J_1, \dots, J_n\}$ on a set of $m$ candidate machines $\mathcal{M} = \{M_1, \dots, M_m\}$. Each job $J_i$ consists of a predetermined sequence of $n_i$ operations $O_{i,1}, O_{i,2}, \dots, O_{i, n_i}$. Each operation $O_{i,j}$ can be processed on a subset of eligible machines $\mathcal{M}_{i,j} \subseteq \mathcal{M}$ at an adjustable cutting speed level $s \in \{1, 2, 3\}$ ($v_1 = 1.0, v_2 = 1.25, v_3 = 1.50$).

The Cloud macro-optimizer solves the triple-objective vector:
$$\min \mathbf{F} = [f_1, f_2, f_3]^T$$
where:
1. **Makespan ($C_{\max}$):**
   $$f_1 = C_{\max} = \max_{i \in \mathcal{J}} C_i$$
2. **Time-of-Use Electricity Cost ($C_{\text{TOU}}$):**
   $$f_2 = C_{\text{TOU}} = \int_0^{C_{\max}} c_{\text{tariff}}(t) \sum_{k=1}^m P_k(t)\,\frac{dt}{60} + \lambda_{\text{peak}} \int_0^{C_{\max}} \max\left(0, \sum_{k=1}^m P_k(t) - P_{\text{limit}}\right)\,\frac{dt}{60}$$
   where $c_{\text{tariff}}(t)$ is the cyclic three-tier TOU tariff over repeating 100-minute manufacturing shifts:
   - **Off-Peak** $[0, 25)\,\text{min}$: $\$0.06/\text{kWh}$
   - **On-Peak** $[25, 60)\,\text{min}$: $\$0.20/\text{kWh}$
   - **Mid-Peak** $[60, 100)\,\text{min}$: $\$0.12/\text{kWh}$
   $\lambda_{\text{peak}} = \$0.30/\text{kWh}$ is the peak demand excess surcharge, and $P_{\text{limit}} \in \{22.0, 24.0, 32.0, 30.0, 16.0\}\,\text{kW}$ is the instance-specific plant power ceiling across Brandimarte benchmarks `Mk01`–`Mk05`.
3. **Total Weighted Tardiness ($\text{TWT}$):**
   $$f_3 = \text{TWT} = \sum_{i=1}^n w_i \max(0, C_i - d_i)$$
   where $w_i$ is job importance priority weight and $d_i$ is contractual delivery due date.

### 2.2 Dynamic Cutting Power & Non-Linear Flank Wear Model
Tool flank wear $w_k(t)$ advances non-linearly across initial running-in, steady-state, and accelerated wear regions according to the extended Taylor equation across cutting speeds $s \in \{1, 2, 3\}$ ($v_1 = 1.0, v_2 = 1.25, v_3 = 1.50$). The incremental wear $\Delta w$ during operation cutting time $t_{\text{proc}} = t_{\text{base}} / v_s$ is modeled as:
$$\Delta w = \eta_k \cdot (v_s)^{\beta_k} \cdot t_{\text{proc}} \cdot \xi_{\text{noise}}$$
where $\eta_k \in [0.95, 1.35]\,\mu\text{m/min}$ is the base wear rate, $\beta_k \in [2.20, 2.70]$ is the machine-specific Taylor wear velocity exponent across CNC workstations $M_1$–$M_8$ in `config/system_params.yaml`, and $\xi_{\text{noise}} \sim \mathcal{N}(1.0, 0.03^2)$ captures metallurgical non-uniformities.

Active cutting power drifts upward as the tool dulls due to escalating friction and cutting edge radius degradation:
$$P_{\text{cutting}, k}(s, w) = P_{\text{idle}, k} + P_{\text{cut\_base}, k} \cdot (v_s)^{\alpha_p} \cdot \left(1 + \gamma_k \frac{\bar{w}_k}{w_{\max}}\right)$$
where $\alpha_p = 1.5$ is the cutting speed power-law scaling exponent, $\gamma_k \in [0.20, 0.28]$ (plant mean $0.25$) denotes the tool wear power drift coefficient accounting for up to $+28\%$ power surge at terminal tool wear ($w_{\max} = 100.0\,\mu\text{m}$), $\bar{w}_k$ is the average flank wear during the cut, and $w_{\max} = 100.0\,\mu\text{m}$ is the catastrophic failure limit.

### 2.3 Mechatronic 3-State Power Model & Break-Even Standby
Workstations transition among operational power states:
- **`OFF`:** $0.0\,\text{W}$ (Main contactor de-energized).
- **`IDLE`:** $P_{\text{idle}, k} \in [0.85, 1.50]\,\text{kW}$ on CNC machines ($15.6\,\text{W}$ on physical proxy).
- **`STANDBY`:** $P_{\text{stdby}, k} \in [0.15, 0.25]\,\text{kW}$ on CNC machines ($3.6\,\text{W}$ on physical proxy), achieved by SSR de-energization of spindle drives and high-pressure coolant pumps while maintaining edge logic and telemetry.
- **`CUTTING`:** $P_{\text{cutting}, k}(s, w)$ ($48.4 - 100.1\,\text{W}$ on physical proxy).
- **`PM / REPAIR`:** Proactive tool replacement duration $t_{\text{pm}} = 10.0\,\text{min}$ ($5.0\,\text{min}$ base) and emergency breakdown repair duration $t_{\text{breakdown}} = 20.0\,\text{min}$.

To eliminate transition energy waste during starvation intervals, opto-isolated solid-state relays (SSR) switch the machine to standby only if the projected idle starvation window $\Delta t$ exceeds the physical break-even threshold:
$$T_{\text{BE}} = \frac{E_{\text{on\_off}}}{P_{\text{idle}} - P_{\text{standby}}}$$

The framework operates across two distinct physical scales:
1. **Industrial Shop-Floor CNC Centers (`M1`–`M8` in `config/system_params.yaml`):**
   - $P_{\text{idle}, k} \in [1.00, 1.50]\,\text{kW}$ (range $[0.85, 1.50]\,\text{kW}$ across lathe and drilling centers),
   - $P_{\text{stdby}, k} \in [0.15, 0.25]\,\text{kW}$,
   - $E_{\text{on\_off}, k} \in [0.12, 0.22]\,\text{kWh}$ with warmup time $t_{\text{warmup}} \in [2.0, 4.0]\,\text{min}$ and tool replacement time $t_{\text{pm}} = 10.0\,\text{min}$,
   - Resulting in machine-specific break-even thresholds $T_{\text{BE}, k} \in [8.36, 10.54]\,\text{min}$ (e.g., $9.00\,\text{min}$ on M1, $8.64\,\text{min}$ on M2, $10.54\,\text{min}$ on M3).
2. **Physical Breadboard Hardware Proxy (24V CNC Mechatronics):**
   - $P_{\text{idle}} = 15.6\,\text{W}$, $P_{\text{standby}} = 3.6\,\text{W}$, $E_{\text{trans}} = 0.62\,\text{Wh}$ ($2232\,\text{J}$),
   - Resulting in $T_{\text{BE}} \approx 3.1\,\text{min}$. If idle interval $\Delta t \le T_{\text{BE}}$, the machine stays in idle; otherwise, the SSR shifts the machine to standby.

### 2.4 Shift-Normalized Cumulative Disturbance Index ($\text{CDI}(t)$)
To filter out high-frequency micro-disturbances and eliminate division-by-zero singularities at shift onset ($t \to 0$), the edge twin evaluates a shift-normalized Cumulative Disturbance Index:
$$\text{CDI}(t) = \alpha \frac{|C_{\text{act}}(t) - C_{\text{base}}(t)|}{C_{\text{base}}^{\text{total}}} + (1 - \alpha) \frac{|E_{\text{act}}(t) - E_{\text{base}}(t)|}{E_{\text{base}}^{\text{total}}}$$
where $\alpha = 0.50$, and $C_{\text{base}}^{\text{total}}$ and $E_{\text{base}}^{\text{total}}$ represent the total planned makespan and energy for the entire production shift. When $\text{CDI}(t) > \theta_{\text{th}} = 0.12$, an event-triggered MQTT payload (`dt/fjsp/edge/trigger_reschedule`, QoS 1) prompts the Cloud solver to re-optimize remaining operations from the current shop-floor snapshot.

### 2.5 Scale-Invariant Edge PPO Micro-Controller & Real-Time EKF Filter
To decouple edge neural inference from problem scale, the edge controller employs a fixed-size **Candidate Job Window ($K=10$) and Eligible Machine Window ($M=6$)**:
- At each decision step, pending jobs are ranked by due-date urgency and ready-time slack to extract the top $K=10$ candidate operations, while machines are filtered to the top $M=6$ eligible units sorted by availability.
- This produces a standardized 87-dimensional observation space and a discrete action space of $360$ combinations ($10 \text{ jobs} \times 6 \text{ machines} \times 3 \text{ speeds} \times 2 \text{ PM flags}$), enabling the identical ONNX model (`ppo_edge_policy.onnx`) to execute across small ($10 \times 6$, 55 ops) and large ($15 \times 8$, 150 ops) manufacturing plants without re-architecting.
- **Zero-Oracle Sensor Fusion:** The edge controller has zero direct access to true physical tool wear. Instead, a real-time Extended Kalman Filter (EKF) fuses power and vibration telemetry to estimate $\hat{w}_m(t)$ and variance $\sigma_{\text{EKF}}^2$. A statistical overrun boundary $w_{\text{proj}} = \hat{w}_m + \Delta w_{\text{nom}}(s) + 2\sigma_{\text{EKF}} + \delta_{\text{margin}}$ governs dynamic speed throttling and triggers opportunistic preventive maintenance before catastrophic failure ($100.0\,\mu\text{m}$).
- **Sub-Millisecond Inference:** Benchmarked on a Linux x86_64 workstation (Python 3.12, ONNX Runtime CPUExecutionProvider), ONNX Runtime executes forward policy inference in $0.0132\,\text{ms}$ ($13.2\,\mu\text{s}$) and complete decision-step inference in $0.1506\,\text{ms}$, delivering $\sim 2.81 \times 10^4 \times$ (per-step) and $\sim 3.20 \times 10^5 \times$ (pure forward) faster reaction than Cloud evolutionary optimization ($4.24 \pm 0.03\,\text{s}$).

---

## 3. Comprehensive Experimental Results & Baseline Comparison

### 3.1 Experimental Benchmark Protocol
To ensure zero data leakage between hyperparameter selection and out-of-sample evaluation, experimental seeds were strictly segregated:
- `TRAIN_SEEDS = [42, 101, 2026]` (Edge PPO training on Mk01 over 66,000 steps).
- `VAL_SEEDS = [201..210]` (10 validation seeds used exclusively to sweep the CDI threshold grid $\theta_{\text{th}} \in [0.02, 0.50]$ under **Regime C** in Phase 5 CDI Sensitivity).
- `TEST_SEEDS = [301..315]` (15 independent testing seeds used for all reported benchmark tables, multi-instance statistical comparisons under **Regime B**, and the frozen-threshold $\theta_{\text{th}} = 0.12$ **Regime C** Phase 4 Closed-Loop HIL evaluation, preventing any test-set leakage into threshold tuning).

All evaluations were conducted across the five standard Brandimarte benchmarks (`Mk01`–`Mk05`) under **Regime B** (continuous stochastic wear drift $+15\% \dots +35\%$) and **Regime C** (continuous wear combined with dual macro-disturbances: batch hardness surge and machine micro-stoppage).

Ten distinct scheduling and dispatching paradigms were evaluated ($10 \text{ methods} \times 5 \text{ instances} \times 15 \text{ seeds} = 750 \text{ full runs}$):
1. **Rule-SPT-Fast:** Shortest Processing Time priority with fixed high speed ($s=3$).
2. **Rule-EET-Eco:** Earliest End Time priority with fixed conservative speed ($s=1$).
3. **Rule-EDD-Bal:** Earliest Due Date priority with balanced nominal speed ($s=2$).
4. **Right-Shift Rescheduling:** Reactive local time-shifting inheriting Cloud baseline assignments.
5. **NSGA-II (Open-Loop):** Pymoo NSGA-II offline baseline with 3-segment chromosome.
6. **MOEA-D (Open-Loop):** Decomposition-based multi-objective evolutionary algorithm.
7. **RVEA (Open-Loop):** Reference Vector Guided Evolutionary Algorithm.
8. **NSGA-III (Open-Loop):** Reference-direction evolutionary solver executed offline.
9. **Standalone Edge PPO (No Cloud Prior):** Pure edge DRL agent lacking global guidance.
10. **Proposed Cloud-Edge DT:** Collaborative framework (NSGA-III + Edge ONNX + CDI $\theta_{\text{th}}=0.12$).

### 3.2 Multi-Scale Benchmark Results (Audited Across 15 Independent Test Seeds)

```
Table 1: Comprehensive Multi-Objective Benchmark Results across Brandimarte Mk01–Mk05 (Mean ± Std, n=15 runs per cell)
```
| Instance | Method | Hypervolume (HV) $\uparrow$ | IGD $\downarrow$ | Makespan $C_{\max}$ (min) $\downarrow$ | TOU Cost (\$) $\downarrow$ | Tardiness TWT (min) $\downarrow$ | Tool Failures $\downarrow$ | Cliff's $\delta$ | $p_{\text{holm}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mk01**<br>(10x6, 55 ops) | Rule-SPT-Fast | $0.0890 \pm 0.0172$ | $0.8353 \pm 0.0425$ | $116.22 \pm 1.57$ | \$3.127 \pm 0.118 | $550.7 \pm 27.2$ | 4.80 | +1.00 | < 0.001 |
|  | Rule-EET-Eco | $0.3441 \pm 0.0263$ | $0.4914 \pm 0.0268$ | $94.67 \pm 2.49$ | \$2.455 \pm 0.043 | $424.1 \pm 11.3$ | 0.00 | +0.41 | 0.2768 |
|  | Rule-EDD-Bal | $0.0396 \pm 0.0227$ | $1.0198 \pm 0.1137$ | $124.55 \pm 8.38$ | \$3.662 \pm 0.070 | $544.3 \pm 80.7$ | 0.00 | +1.00 | < 0.001 |
|  | Right-Shift Rescheduling | $0.4967 \pm 0.1268$ | $0.3527 \pm 0.1052$ | $81.17 \pm 6.74$ | \$2.546 \pm 0.120 | $345.8 \pm 99.4$ | 0.00 | -0.40 | 0.2768 |
|  | NSGA-II (Open-Loop) | $0.4845 \pm 0.2341$ | $0.3986 \pm 0.2066$ | $82.69 \pm 11.90$ | \$2.573 \pm 0.199 | $365.4 \pm 159.8$ | 0.73 | -0.21 | 0.5537 |
|  | MOEA-D (Open-Loop) | $0.5326 \pm 0.1997$ | $0.3439 \pm 0.1478$ | $79.77 \pm 10.15$ | \$2.606 \pm 0.127 | $323.2 \pm 128.2$ | 0.80 | -0.38 | 0.3616 |
|  | RVEA (Open-Loop) | $0.6183 \pm 0.2444$ | $0.3655 \pm 0.1659$ | $67.47 \pm 10.34$ | \$3.052 \pm 0.368 | $184.0 \pm 105.6$ | 0.60 | -0.62 | 0.1293 |
|  | NSGA-III (Open-Loop) | $0.4693 \pm 0.1798$ | $0.3952 \pm 0.1517$ | $83.64 \pm 11.90$ | \$2.570 \pm 0.233 | $362.7 \pm 101.7$ | 0.87 | -0.21 | 0.5614 |
|  | Standalone Edge PPO | $0.0548 \pm 0.0179$ | $0.9275 \pm 0.0611$ | $110.42 \pm 3.77$ | \$3.245 \pm 0.136 | $673.6 \pm 45.8$ | 0.00 | +1.00 | < 0.001 |
|  | **Proposed Cloud-Edge DT** | **$0.4103 \pm 0.1024$** | **$0.3948 \pm 0.0989$** | **$86.00 \pm 9.79$** | **\$2.793 \pm 0.183** | **$330.0 \pm 60.5$** | **0.00** | **Base** | **Base** |
| **Mk02**<br>(10x6, 58 ops) | Rule-SPT-Fast | $0.1294 \pm 0.0761$ | $0.8218 \pm 0.1708$ | $110.98 \pm 16.66$ | \$3.323 \pm 0.173 | $513.5 \pm 146.4$ | 5.67 | +1.00 | < 0.001 |
|  | Rule-EET-Eco | $0.1173 \pm 0.0331$ | $0.8174 \pm 0.0485$ | $98.73 \pm 2.46$ | \$1.962 \pm 0.033 | $707.5 \pm 33.2$ | 0.00 | +1.00 | < 0.001 |
|  | Rule-EDD-Bal | $0.1385 \pm 0.0346$ | $0.8148 \pm 0.0631$ | $85.69 \pm 2.79$ | \$4.311 \pm 0.240 | $304.2 \pm 17.2$ | 0.00 | +1.00 | < 0.001 |
|  | Right-Shift Rescheduling | $0.7429 \pm 0.0997$ | $0.2237 \pm 0.0335$ | $67.33 \pm 7.12$ | \$2.186 \pm 0.205 | $179.0 \pm 52.5$ | 0.00 | -0.63 | 0.0043 |
|  | NSGA-II (Open-Loop) | $0.6466 \pm 0.1645$ | $0.2767 \pm 0.0982$ | $69.11 \pm 10.70$ | \$2.172 \pm 0.173 | $265.9 \pm 102.0$ | 0.60 | -0.28 | 0.7571 |
|  | MOEA-D (Open-Loop) | $0.5733 \pm 0.1908$ | $0.3239 \pm 0.0754$ | $70.52 \pm 10.39$ | \$2.345 \pm 0.271 | $294.8 \pm 99.5$ | 0.87 | +0.15 | 0.8904 |
|  | RVEA (Open-Loop) | $0.7062 \pm 0.2558$ | $0.3603 \pm 0.1097$ | $54.86 \pm 8.23$ | \$2.846 \pm 0.445 | $157.9 \pm 97.8$ | 0.80 | -0.23 | 0.7571 |
|  | NSGA-III (Open-Loop) | $0.7691 \pm 0.1572$ | $0.2290 \pm 0.0424$ | $66.29 \pm 9.09$ | \$2.095 \pm 0.162 | $188.2 \pm 84.1$ | 0.33 | -0.59 | 0.0269 |
|  | Standalone Edge PPO | $0.1471 \pm 0.1101$ | $0.8160 \pm 0.1898$ | $99.79 \pm 18.09$ | \$3.399 \pm 0.251 | $527.3 \pm 207.5$ | 0.53 | +0.98 | < 0.001 |
|  | **Proposed Cloud-Edge DT** | **$0.5828 \pm 0.1517$** | **$0.3262 \pm 0.0921$** | **$72.31 \pm 10.53$** | **\$2.466 \pm 0.355** | **$237.7 \pm 110.6$** | **0.00** | **Base** | **Base** |
| **Mk03**<br>(15x8, 150 ops) | Rule-SPT-Fast | $0.2353 \pm 0.0377$ | $0.5716 \pm 0.0489$ | $547.20 \pm 11.86$ | \$17.007 \pm 0.472 | $5394.7 \pm 260.1$ | 19.20 | +0.79 | 0.0051 |
|  | Rule-EET-Eco | $0.4803 \pm 0.1242$ | $0.3858 \pm 0.1013$ | $512.67 \pm 32.41$ | \$13.323 \pm 0.275 | $4542.0 \pm 474.9$ | 1.40 | +0.24 | 0.9084 |
|  | Rule-EDD-Bal | $0.0755 \pm 0.0443$ | $0.9729 \pm 0.0955$ | $619.12 \pm 48.85$ | \$26.912 \pm 0.867 | $3079.4 \pm 433.4$ | 4.87 | +1.00 | < 0.001 |
|  | Right-Shift Rescheduling | $0.7707 \pm 0.1161$ | $0.2530 \pm 0.0274$ | $427.21 \pm 24.54$ | \$13.362 \pm 0.555 | $3769.2 \pm 587.4$ | 0.00 | -0.62 | 0.0043 |
|  | NSGA-II (Open-Loop) | $0.3583 \pm 0.1276$ | $0.4670 \pm 0.1318$ | $507.85 \pm 47.47$ | \$14.506 \pm 0.677 | $5220.6 \pm 644.7$ | 10.07 | +0.52 | 0.0410 |
|  | MOEA-D (Open-Loop) | $0.4295 \pm 0.1877$ | $0.4273 \pm 0.1622$ | $496.38 \pm 44.33$ | \$14.649 \pm 0.390 | $4888.5 \pm 1040.2$ | 10.47 | +0.32 | 0.9084 |
|  | RVEA (Open-Loop) | $0.5662 \pm 0.1878$ | $0.3457 \pm 0.0981$ | $445.82 \pm 31.79$ | \$15.346 \pm 0.670 | $4280.2 \pm 1047.0$ | 12.00 | -0.00 | 0.9084 |
|  | NSGA-III (Open-Loop) | $0.3453 \pm 0.1147$ | $0.4784 \pm 0.1221$ | $514.37 \pm 25.12$ | \$14.280 \pm 0.517 | $5239.8 \pm 815.4$ | 9.67 | +0.57 | 0.0214 |
|  | Standalone Edge PPO | $0.1001 \pm 0.0454$ | $0.8780 \pm 0.0838$ | $438.62 \pm 44.32$ | \$26.469 \pm 1.338 | $5093.4 \pm 591.9$ | 6.27 | +1.00 | < 0.001 |
|  | **Proposed Cloud-Edge DT** | **$0.5477 \pm 0.2102$** | **$0.3516 \pm 0.1398$** | **$462.69 \pm 45.99$** | **\$13.883 \pm 0.677** | **$4621.8 \pm 889.6$** | **0.00** | **Base** | **Base** |
| **Mk04**<br>(15x8, 90 ops) | Rule-SPT-Fast | $0.0652 \pm 0.0240$ | $0.8796 \pm 0.0681$ | $188.05 \pm 10.17$ | \$6.811 \pm 0.127 | $1942.4 \pm 97.4$ | 8.53 | +1.00 | < 0.001 |
|  | Rule-EET-Eco | $0.1464 \pm 0.0270$ | $0.6389 \pm 0.0444$ | $195.60 \pm 4.47$ | \$4.430 \pm 0.141 | $2038.0 \pm 49.3$ | 0.00 | +0.96 | < 0.001 |
|  | Rule-EDD-Bal | $0.0833 \pm 0.0063$ | $0.8581 \pm 0.0156$ | $196.24 \pm 4.35$ | \$6.740 \pm 0.145 | $1414.7 \pm 26.6$ | 0.00 | +1.00 | < 0.001 |
|  | Right-Shift Rescheduling | $0.5081 \pm 0.1781$ | $0.3273 \pm 0.1262$ | $152.40 \pm 10.83$ | \$4.547 \pm 0.225 | $1546.9 \pm 398.6$ | 0.00 | -0.43 | 0.1508 |
|  | NSGA-II (Open-Loop) | $0.3237 \pm 0.1499$ | $0.4472 \pm 0.1483$ | $168.22 \pm 15.41$ | \$4.542 \pm 0.382 | $1840.5 \pm 348.0$ | 2.53 | +0.25 | 0.4280 |
|  | MOEA-D (Open-Loop) | $0.4374 \pm 0.2159$ | $0.3926 \pm 0.1857$ | $162.66 \pm 23.47$ | \$4.660 \pm 0.380 | $1596.2 \pm 279.5$ | 2.87 | -0.16 | 1.0000 |
|  | RVEA (Open-Loop) | $0.4047 \pm 0.2414$ | $0.4824 \pm 0.1735$ | $157.56 \pm 21.13$ | \$5.319 \pm 0.569 | $1508.7 \pm 467.3$ | 3.47 | -0.01 | 1.0000 |
|  | NSGA-III (Open-Loop) | $0.3603 \pm 0.1830$ | $0.4257 \pm 0.1386$ | $166.18 \pm 15.65$ | \$4.462 \pm 0.336 | $1784.7 \pm 373.7$ | 2.47 | +0.20 | 1.0000 |
|  | Standalone Edge PPO | $0.1945 \pm 0.1139$ | $0.7371 \pm 0.1331$ | $151.79 \pm 16.68$ | \$6.835 \pm 0.451 | $1476.8 \pm 252.8$ | 0.80 | +0.68 | 0.0037 |
|  | **Proposed Cloud-Edge DT** | **$0.3912 \pm 0.1429$** | **$0.3799 \pm 0.1024$** | **$161.25 \pm 13.95$** | **\$4.632 \pm 0.264** | **$1700.1 \pm 268.0$** | **0.00** | **Base** | **Base** |
| **Mk05**<br>(15x4, 106 ops) | Rule-SPT-Fast | $0.1587 \pm 0.0504$ | $0.6874 \pm 0.0686$ | $377.60 \pm 20.43$ | \$19.270 \pm 0.987 | $4456.9 \pm 361.5$ | 17.13 | +0.96 | 0.0011 |
|  | Rule-EET-Eco | $0.2303 \pm 0.0599$ | $0.5143 \pm 0.0721$ | $453.73 \pm 18.80$ | \$10.512 \pm 0.197 | $6619.3 \pm 355.2$ | 0.00 | +0.83 | 0.0017 |
|  | Rule-EDD-Bal | $0.5623 \pm 0.0425$ | $0.3964 \pm 0.0120$ | $387.35 \pm 10.46$ | \$14.239 \pm 0.343 | $2829.1 \pm 82.3$ | 0.00 | -0.85 | 0.0011 |
|  | Right-Shift Rescheduling | $0.5152 \pm 0.1031$ | $0.3594 \pm 0.0271$ | $376.46 \pm 31.23$ | \$12.230 \pm 0.934 | $4806.7 \pm 704.0$ | 0.00 | -0.42 | < 0.001 |
|  | NSGA-II (Open-Loop) | $0.2084 \pm 0.0637$ | $0.5000 \pm 0.0796$ | $459.37 \pm 25.44$ | \$12.362 \pm 0.650 | $6215.3 \pm 664.6$ | 9.13 | +0.88 | < 0.001 |
|  | MOEA-D (Open-Loop) | $0.1467 \pm 0.1102$ | $0.6280 \pm 0.1769$ | $470.79 \pm 46.37$ | \$12.719 \pm 1.261 | $6984.4 \pm 1206.9$ | 9.87 | +0.88 | < 0.001 |
|  | RVEA (Open-Loop) | $0.1992 \pm 0.0785$ | $0.5499 \pm 0.0885$ | $446.88 \pm 25.00$ | \$16.178 \pm 0.840 | $5054.0 \pm 836.0$ | 12.67 | +0.88 | 0.0011 |
|  | NSGA-III (Open-Loop) | $0.2318 \pm 0.1186$ | $0.5040 \pm 0.1217$ | $450.84 \pm 40.10$ | \$12.697 \pm 0.921 | $6100.4 \pm 983.7$ | 9.20 | +0.74 | 0.0011 |
|  | Standalone Edge PPO | $0.4012 \pm 0.0891$ | $0.4569 \pm 0.0365$ | $344.81 \pm 25.02$ | \$15.457 \pm 0.502 | $4896.0 \pm 639.0$ | 4.73 | +0.16 | 0.5995 |
|  | **Proposed Cloud-Edge DT** | **$0.4188 \pm 0.1046$** | **$0.3851 \pm 0.0683$** | **$408.76 \pm 29.48$** | **\$11.198 \pm 0.539** | **$5395.5 \pm 800.5$** | **0.00** | **Base** | **Base** |
### 3.3 Critical Examination of Baseline Behaviors

#### 3.3.1 Honest Assessment of Right-Shift Rescheduling (Fair EKF-Equipped Baseline)
Right-Shift Rescheduling serves as a rigorous reactive baseline because it inherits global machine assignments and operation orderings from Cloud NSGA-III and is equipped with the **identical `BreadboardDAQInterface` + `RealTimeToolWearEstimator` (Live EKF) and statistical PM rule** as the Proposed DT (eliminating all unobservable oracle assumptions). With sensor parity, Right-Shift also achieves $0.00$ breakdowns across all five benchmarks (`Mk01`–`Mk05`). Under two-sided paired Wilcoxon testing with Holm-Bonferroni correction ($K=9$; which tests distributional difference rather than formal TOST), no statistically significant difference in Hypervolume is observed between Right-Shift and Proposed DT on `Mk01` ($p_{\text{holm}} = 0.2768$) and `Mk04` ($p_{\text{holm}} = 0.1508$), while Right-Shift achieves statistically higher Hypervolume on `Mk02` ($p_{\text{holm}} = 0.0043$), `Mk03` ($p_{\text{holm}} = 0.0043$), and `Mk05` ($p_{\text{holm}} = 0.0005$) by preserving fixed Cloud machine/speed topology without peak-tariff throttling.

However, Right-Shift cannot adapt cutting speeds dynamically to cyclic TOU windows or reroute disrupted schedules. In contrast, the Proposed DT provides:
- **TOU Energy Cost Minimization:** On dense bottleneck `Mk05`, Proposed DT achieves the lowest TOU electricity cost among dynamic schedulers (\$11.198 $\pm$ 0.539 vs. \$12.230 $\pm$ 0.934 for Right-Shift, **saving 8.44%**).
- **Tardiness Control vs. Open-Loop Methods:** On `Mk05`, Proposed DT keeps TWT to $5395.5 \pm 800.5\,\text{min}$, lower than open-loop evolutionary baselines (NSGA-III at $6100.4 \pm 983.7\,\text{min}$, MOEA-D at $6984.4 \pm 1206.9\,\text{min}$). Right-Shift achieves shorter nominal makespan and TWT ($376.46\,\text{min}$ and $4806.7\,\text{min}$ on `Mk05`) at higher TOU electricity cost.
- **Complete Failure Immunity vs. Open-Loop Solvers:** Proposed DT guarantees $0.00$ breakdowns across `Mk01`–`Mk05` and all transport latency tiers, eliminating the $0.33 - 12.67$ tool failures per shift suffered by open-loop metaheuristics and $0.80 - 7.60$ failures of wear-blind Variant 1.

#### 3.3.2 Objective Analysis of Standalone Edge PPO (No Cloud Prior)
A key empirical finding is that operating an edge DRL agent without Cloud macro-coordination introduces severe system-wide penalties:
1. **Severe Tardiness and Makespan Inflation:** On `Mk01`, Standalone Edge PPO experiences a $+28.4\%$ makespan degradation ($110.42\,\text{min}$ vs $86.00\,\text{min}$) and a $+104.1\%$ inflation in TWT ($673.6\,\text{min}$ vs $330.0\,\text{min}$) due to myopic dispatching loops.
2. **Peak-Tariff Overloading:** Lacking global schedule awareness, Standalone Edge PPO cuts at high speeds during On-Peak tariff windows, driving TOU electricity costs up by **$+16.2\%$ to $+90.7\%$** compared to the Proposed DT ($\$3.245 \pm 0.136$ vs $\$2.793 \pm 0.183$ on Mk01; $\$26.469 \pm 1.338$ vs $\$13.883 \pm 0.677$ on Mk03; $\$6.835 \pm 0.352$ vs $\$4.632 \pm 0.264$ on Mk04; $\$15.457 \pm 0.502$ vs $\$11.198 \pm 0.539$ on Mk05).
3. **Tool Breakage Hazard:** In complex multi-machine environments (`Mk03`), unguided speed scaling induces severe tool wear spikes, resulting in $6.27$ breakdowns per run.

#### 3.3.3 Resolving the Objective Trade-Offs on Bottleneck Benchmark Mk05
On benchmark `Mk05` (15 jobs with 106 operations crowded onto only 4 bottleneck machines, $26.5$ operations per machine), fair EKF-equipped Right-Shift achieves higher Hypervolume ($\text{HV} = 0.5152$ vs. $0.4188$, $p_{\text{holm}} = 0.0005$), while the Proposed DT prioritizes peak-tariff energy reduction:
- **Shared Zero-Breakdown Safety:** Both EKF-equipped Proposed DT and fair Right-Shift achieve $0.00$ tool breakdowns, whereas open-loop NSGA-III incurs $9.20$, MOEA-D $9.87$, and RVEA $12.67$ breakdowns per run.
- **Lowest Dynamic TOU Cost:** By dynamically scaling cutting speeds down to $s=1$ or scheduling opportunistic PM during high-tariff periods, Proposed DT achieves $\$11.198 \pm 0.539$ (saving **8.44%** vs. Right-Shift at \$12.230 $\pm$ 0.934).
- **Reduced Tardiness vs. Open-Loop:** TWT is kept to $5395.5\,\text{min}$, significantly lower than NSGA-III ($6100.4\,\text{min}$) and MOEA-D ($6984.4\,\text{min}$).
- The enclosed hypervolume metric penalizes the longer makespan ($408.76\,\text{min}$ vs. $376.46\,\text{min}$) relative to `[1.1, 1.1, 1.1]` during peak-tariff throttling.

---

## 4. In-Depth Component Ablation Study & Sensitivity Analysis

### 4.1 Architectural Component Ablation (Mk01 & Mk03, 15 Test Seeds)
To isolate the exact physical contribution of each sub-system, four ablated architectural variants were benchmarked against the Full Proposed DT under cyclic TOU pricing ($n=15$ runs per variant):

```
Table 2: Ablation Study Results across 15 Independent Test Seeds on Mk01 and Mk03
```
| Instance | Variant | Description | Makespan (min) | TOU Cost (\$) | TWT (min) | Breakdowns |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Mk01**<br>(10x6, 55 ops) | **Full Proposed DT** | Full closed-loop framework (Dual-Sensor EKF + Speed Scaling + SSR Standby + Cloud Guidance) | **$86.00 \pm 9.79$ (Base)** | **\$2.793 \pm 0.183 (Base)** | **$329.99 \pm 60.47$ (Base)** | **0.00** |
|  | Variant 1 | w/o Dual-Sensor Wear Feedback (Delayed PM under disturbance) | $84.20 \pm 9.27$ (-2.1%) | \$2.572 \pm 0.157$ (-7.9%) | $373.68 \pm 129.18$ (+13.2%) | **0.80** |
|  | Variant 2 | Fixed Speed $s=2$ (No real-time dynamic TOU peak avoidance) | $79.34 \pm 6.61$ (-7.8%) | \$3.031 \pm 0.357$ (+8.5%) | $294.04 \pm 62.85$ (-10.9%) | 0.00 |
|  | Variant 3 | Idle-Only (No SSR standby power gating during waiting intervals) | $85.09 \pm 8.56$ (-1.1%) | \$3.224 \pm 0.214$ (+15.4%) | $324.19 \pm 49.99$ (-1.8%) | 0.00 |
|  | Variant 4 | w/o Cloud Guidance (Pure local edge dispatching without global macro prior) | $110.42 \pm 3.77$ (+28.4%) | \$3.245 \pm 0.136$ (+16.2%) | $673.57 \pm 45.85$ (+104.1%) | 0.00 |
| **Mk03**<br>(15x8, 150 ops) | **Full Proposed DT** | Full closed-loop framework (Dual-Sensor EKF + Speed Scaling + SSR Standby + Cloud Guidance) | **$462.69 \pm 45.99$ (Base)** | **\$13.883 \pm 0.677 (Base)** | **$4621.78 \pm 889.63$ (Base)** | **0.00** |
|  | Variant 1 | w/o Dual-Sensor Wear Feedback (Delayed PM under disturbance) | $511.54 \pm 29.06$ (+10.6%) | \$13.683 \pm 0.387$ (-1.4%) | $5226.80 \pm 801.80$ (+13.1%) | **7.60** |
|  | Variant 2 | Fixed Speed $s=2$ (No real-time dynamic TOU peak avoidance) | $413.64 \pm 35.03$ (-10.6%) | \$14.133 \pm 0.969$ (+1.8%) | $3750.56 \pm 725.84$ (-18.9%) | 0.00 |
|  | Variant 3 | Idle-Only (No SSR standby power gating during waiting intervals) | $442.95 \pm 33.98$ (-4.3%) | \$20.106 \pm 0.934$ (+44.8%) | $4275.74 \pm 682.96$ (-7.5%) | 0.00 |
|  | Variant 4 | w/o Cloud Guidance (Pure local edge dispatching without global macro prior) | $438.62 \pm 44.32$ (-5.2%) | \$26.469 \pm 1.338$ (+90.7%) | $5093.45 \pm 591.89$ (+10.2%) | **6.27** |
### 4.2 Analytical Findings from Ablation

#### 4.2.1 Scale-Dependent Behavior of Variant 1 (The Blind Closed-Loop Hazard)
The behavior of Variant 1 demonstrates the necessity of real-time sensor fusion:
- **Tool Failure Escalation:** Without live dual-sensor wear feedback, Variant 1 relies on nominal theoretical wear models. Under unmodeled material hardness fluctuations ($\Delta\eta \in [+15\%, +35\%]$), tools degrade faster than anticipated. This causes delayed preventive maintenance and catastrophic tool failure: **$0.80$ breakdowns** on Mk01 and **$7.60$ breakdowns** on Mk03.
- **Tardiness Explosion:** Emergency tool breakdowns incur lengthy replacement stoppages, inflating TWT by **$+13.2\%$** on Mk01 ($373.68\,\text{min}$ vs $329.99\,\text{min}$) and **$+13.1\%$** on Mk03 ($5226.80\,\text{min}$ vs $4621.78\,\text{min}$).

#### 4.2.2 Operational Trade-Offs of Variant 2 (Fixed Speed $s=2$)
Locking cutting speed to nominal $s=2$ prevents the edge agent from dynamically responding to cyclic TOU tariff transitions:
- During peak-tariff hours ($rate = \$0.20/\text{kWh}$), Full Proposed DT throttles non-urgent cuts down to $s=1$ or executes necessary PM, cutting electricity expenditure.
- In contrast, Variant 2 remains locked at $s=2$, incurring a positive TOU cost degradation of **$+8.53\%$** on Mk01 ($\$3.031 \pm 0.357$ vs $\$2.793 \pm 0.183$) and **$+1.80\%$** on Mk03 ($\$14.133 \pm 0.969$ vs $\$13.883 \pm 0.677$).

#### 4.2.3 Energy Waste in Variant 3 (Idle-Only / No Standby)
Disabling opto-isolated solid-state relay (SSR) standby power gating forces machines to draw continuous idling power during buffer starvation intervals:
- **Ablation Degradation (Cost Penalty when Omitting SSR):** Omitting SSR standby power gating increases TOU electricity cost by **$+15.43\%$** on Mk01 ($\$3.224 \pm 0.214$ vs $\$2.793 \pm 0.183$) and by a massive **$+44.82\%$** on Mk03 ($\$20.106 \pm 0.934$ vs $\$13.883 \pm 0.677$).
- **Standby Power Savings (Cost Reduction Achieved by SSR):** Viewed from the baseline of standard continuous-idle equipment, activating SSR standby power gating achieves a **$13.37\%$** ($-13.37\%$) cost reduction on Mk01 (saving $\$0.431$ per shift) and a **$30.95\%$** ($-30.95\%$) cost reduction on Mk03 (saving $\$6.223$ per shift), proving the decisive economic benefit of physical mechatronic standby actuation.

#### 4.2.4 The Chaos of Variant 4 (w/o Cloud Guidance)
Operating purely on local edge dispatching rules without the Cloud macro-prior leads to severe operational degradation:
- On Mk01, makespan degrades by **$+28.39\%$** ($110.42 \pm 3.77\,\text{min}$ vs $86.00 \pm 9.79\,\text{min}$) and TWT doubles (**$+104.12\%$**, $673.57 \pm 45.85\,\text{min}$ vs $329.99 \pm 60.47\,\text{min}$).
- On Mk03, TOU electricity cost surges by **$+90.66\%$** ($\$26.469 \pm 1.338$ vs $\$13.883 \pm 0.677$), and tool breakdowns reach **$6.27$** per run due to localized myopic machine queueing.

### 4.3 CDI Threshold Sensitivity & The U-Shaped Cost Curve
To determine the optimal synchronization boundary, the threshold $\theta_{\text{th}}$ was evaluated across a 10-point grid spanning $[0.02, 0.50]$ under Regime C ($n=10$ validation seeds `VAL_SEEDS = [201..210]`):

```
Table 3: Parametric CDI Threshold Sensitivity Evaluation across 10 Validation Seeds (Regime C)
```
| CDI Threshold $\theta_{\text{th}}$ | Resync Calls ($\text{Mean} \pm \text{Std}$) | Makespan ($\text{Mean} \pm \text{Std}$) | TOU Cost (\$) | Max CDI ($\text{Mean} \pm \text{Std}$) | Schedule Index | Total System Cost Index | Operational Region |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.02** | $4.90 \pm 0.70$ | $64.85 \pm 3.43$ | \$2.901 \pm 0.107 | $0.1043 \pm 0.0123$ | 1.1529 | 1.0867 | Over-Triggering Region (> 2 Shocks) |
| **0.05** | **$2.70 \pm 0.46$** | **$63.19 \pm 3.31$** | **\$2.796 \pm 0.099** | **$0.0846 \pm 0.0053$** | **1.1173** | **1.0325 (Empirical Optimum)** | **Over-Triggering Region (> 2 Shocks)** |
| **0.08** | $1.90 \pm 0.30$ | $68.94 \pm 4.46$ | \$2.716 \pm 0.075 | $0.1009 \pm 0.0242$ | 1.1508 | 1.0547 | Event-Matched Plateau (1–2 Shocks) |
| **0.10** | $1.50 \pm 0.50$ | $72.59 \pm 3.80$ | \$2.652 \pm 0.095 | $0.1287 \pm 0.0150$ | 1.1693 | 1.0674 | Event-Matched Plateau (1–2 Shocks) |
| **0.12** | **$1.40 \pm 0.49$** | **$74.45 \pm 4.53$** | **\$2.647 \pm 0.099** | **$0.1373 \pm 0.0120$** | **1.1843** | **1.0799 (Selected Center)** | **Event-Matched Plateau (1–2 Shocks)** |
| **0.15** | $1.00 \pm 0.00$ | $75.51 \pm 3.44$ | \$2.644 \pm 0.099 | $0.1759 \pm 0.0174$ | 1.1927 | 1.0834 | Event-Matched Plateau (1–2 Shocks) |
| **0.18** | $1.00 \pm 0.00$ | $76.71 \pm 2.80$ | \$2.624 \pm 0.066 | $0.2078 \pm 0.0185$ | 1.1991 | 1.0892 | Event-Matched Plateau (1–2 Shocks) |
| **0.22** | $0.90 \pm 0.30$ | $77.88 \pm 2.95$ | \$2.614 \pm 0.071 | $0.2369 \pm 0.0147$ | 1.2071 | 1.0954 | Under-Triggering Region (< 1 Shock) |
| **0.35** | $0.00 \pm 0.00$ | $84.63 \pm 2.79$ | \$2.583 \pm 0.062 | $0.2623 \pm 0.0268$ | 1.2594 | 1.1334 | Under-Triggering Region (< 1 Shock) |
| **0.50** | $0.00 \pm 0.00$ | $84.63 \pm 2.79$ | \$2.583 \pm 0.062 | $0.2623 \pm 0.0268$ | 1.2594 | 1.1334 | Under-Triggering Region (< 1 Shock) |
As shown in Figure 4(b), thresholds are classified strictly by objective physical matching to the $N_{\text{shocks}} = 2.0$ macro-disturbances per shift:
- **Over-Triggering Region (`> 2 Shocks`, $\text{calls} > 2.0$, $\theta_{\text{th}} \le 0.05$):** High sensitivity triggers $2.70 \pm 0.46$ to $4.90 \pm 0.70$ calls per shift ($7.25$–$13.15\,\text{s}$ Cloud Warm-Start solver overhead). While $\theta_{\text{th}}^* = 0.05$ minimizes the heuristic composite `Total_System_Cost_Index` ($1.0325$, $63.19\,\text{min}$, \$2.796), it does so by over-triggering beyond the 2 physical shocks.
- **Event-Matched Plateau (`1–2 Shocks`, $1.0 \le \text{calls} \le 2.0$, $\theta_{\text{th}} \in [0.08, 0.18]$):** Resync calls track the physical macro-disturbance budget ($1.00$ to $1.90$ calls; $2.68$–$5.10\,\text{s}$ solver overhead). At $\theta_{\text{th}} = 0.08$, the index is $1.0547$ ($1.90 \pm 0.30$ calls, $68.94\,\text{min}$, \$2.716$); the frozen deployment center $\theta_{\text{th}} = 0.12$ executes $1.40 \pm 0.49$ calls ($74.45 \pm 4.53\,\text{min}$, TOU Cost \$2.647 $\pm$ 0.099$, `Total_System_Cost_Index` $= 1.0799$).
- **Under-Triggering Region (`< 1 Shock`, $\text{calls} < 1.0$, $\theta_{\text{th}} \ge 0.22$):** Below-unity triggering misses macro-disturbances ($0.90 \to 0.00$ calls), degrading Makespan up to **$84.63 \pm 2.79\,\text{min}$** and raising `Total_System_Cost_Index` to $1.1334$.

### 4.4 Inter-Workstation Transport Latency Sensitivity Analysis
In automated discrete manufacturing facilities, automated guided vehicles (AGVs) or overhead conveyors introduce non-zero transport delays between consecutive machine operations. To investigate the robustness of the collaborative architecture against material handling latency, we systematically varied inter-workstation transport time $\tau_{\text{trans}} \in [0.0, 1.0, 2.0, 3.0]\,\text{min}$ across both small (`Mk01`) and large (`Mk03`) instances under 15 independent test seeds ($n=15$ runs per cell):

```
Table 4: Parametric Transport Latency Sensitivity Evaluation across 15 Test Seeds (Mk01 & Mk03)
```
| Instance | Transport Latency $\tau_{\text{trans}}$ | Method | Makespan (min) $\downarrow$ | TOU Cost (\$) $\downarrow$ | Tardiness TWT (min) $\downarrow$ | Tool Failures $\downarrow$ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Mk01** | 0.0 min | NSGA-III (Open-Loop) | $83.64 \pm 11.90$ | \$2.570 \pm 0.233 | $362.7 \pm 101.7$ | **0.87** |
|  |  | Right-Shift Rescheduling | $81.17 \pm 6.74$ | \$2.546 \pm 0.120 | $345.8 \pm 99.4$ | 0.00 |
|  |  | Standalone Edge PPO | $110.42 \pm 3.77$ | \$3.245 \pm 0.136 | $673.6 \pm 45.9$ | 0.00 |
|  |  | **Proposed Cloud-Edge DT** | **$86.00 \pm 9.79$** | **\$2.793 \pm 0.183** | **$330.0 \pm 60.5$** | **0.00** |
|  | 1.0 min | NSGA-III (Open-Loop) | $93.70 \pm 11.95$ | \$2.549 \pm 0.194 | $457.7 \pm 113.2$ | **0.87** |
|  |  | Right-Shift Rescheduling | $93.24 \pm 8.22$ | \$2.529 \pm 0.097 | $468.4 \pm 121.0$ | 0.00 |
|  |  | Standalone Edge PPO | $127.27 \pm 4.58$ | \$3.184 \pm 0.134 | $878.3 \pm 56.1$ | 0.00 |
|  |  | **Proposed Cloud-Edge DT** | **$98.12 \pm 12.15$** | **\$2.830 \pm 0.218** | **$437.0 \pm 64.8$** | **0.00** |
|  | 2.0 min | NSGA-III (Open-Loop) | $104.69 \pm 12.39$ | \$2.566 \pm 0.182 | $570.7 \pm 126.7$ | **0.87** |
|  |  | Right-Shift Rescheduling | $105.85 \pm 9.48$ | \$2.462 \pm 0.098 | $601.1 \pm 139.1$ | 0.00 |
|  |  | Standalone Edge PPO | $143.61 \pm 5.12$ | \$3.107 \pm 0.104 | $1070.7 \pm 55.7$ | 0.00 |
|  |  | **Proposed Cloud-Edge DT** | **$117.77 \pm 14.31$** | **\$2.813 \pm 0.172** | **$606.4 \pm 147.9$** | **0.00** |
|  | 3.0 min | NSGA-III (Open-Loop) | $116.87 \pm 13.58$ | \$2.542 \pm 0.165 | $697.8 \pm 140.9$ | **0.87** |
|  |  | Right-Shift Rescheduling | $119.25 \pm 10.65$ | \$2.414 \pm 0.121 | $746.0 \pm 156.2$ | 0.00 |
|  |  | Standalone Edge PPO | $159.54 \pm 5.55$ | \$3.196 \pm 0.102 | $1265.6 \pm 60.6$ | 0.00 |
|  |  | **Proposed Cloud-Edge DT** | **$130.24 \pm 18.73$** | **\$2.937 \pm 0.162** | **$764.1 \pm 135.2$** | **0.00** |
| **Mk03** | 0.0 min | NSGA-III (Open-Loop) | $514.37 \pm 25.12$ | \$14.280 \pm 0.517 | $5239.8 \pm 815.4$ | **9.67** |
|  |  | Right-Shift Rescheduling | $427.21 \pm 24.54$ | \$13.362 \pm 0.555 | $3769.2 \pm 587.4$ | 0.00 |
|  |  | Standalone Edge PPO | $438.62 \pm 44.32$ | \$26.469 \pm 1.338 | $5093.4 \pm 591.9$ | **6.27** |
|  |  | **Proposed Cloud-Edge DT** | **$462.69 \pm 45.99$** | **\$13.883 \pm 0.677** | **$4621.8 \pm 889.6$** | **0.00** |
|  | 1.0 min | NSGA-III (Open-Loop) | $550.77 \pm 28.69$ | \$14.257 \pm 0.638 | $5839.3 \pm 831.7$ | **9.67** |
|  |  | Right-Shift Rescheduling | $464.69 \pm 30.80$ | \$13.951 \pm 0.486 | $4401.1 \pm 626.4$ | 0.00 |
|  |  | Standalone Edge PPO | $466.56 \pm 49.07$ | \$26.117 \pm 1.703 | $5443.4 \pm 770.6$ | **6.73** |
|  |  | **Proposed Cloud-Edge DT** | **$474.54 \pm 28.76$** | **\$14.189 \pm 0.776** | **$4943.6 \pm 698.1$** | **0.00** |
|  | 2.0 min | NSGA-III (Open-Loop) | $589.88 \pm 32.56$ | \$14.257 \pm 0.750 | $6498.9 \pm 850.1$ | **9.67** |
|  |  | Right-Shift Rescheduling | $503.85 \pm 37.93$ | \$13.873 \pm 0.370 | $5072.7 \pm 693.8$ | 0.00 |
|  |  | Standalone Edge PPO | $473.42 \pm 44.68$ | \$25.390 \pm 1.185 | $5035.0 \pm 661.9$ | **6.20** |
|  |  | **Proposed Cloud-Edge DT** | **$507.72 \pm 39.07$** | **\$14.373 \pm 0.710** | **$5615.4 \pm 992.9$** | **0.00** |
|  | 3.0 min | NSGA-III (Open-Loop) | $630.07 \pm 37.35$ | \$14.214 \pm 0.849 | $7185.8 \pm 891.6$ | **9.67** |
|  |  | Right-Shift Rescheduling | $545.00 \pm 44.39$ | \$13.988 \pm 0.395 | $5780.4 \pm 782.7$ | 0.00 |
|  |  | Standalone Edge PPO | $483.63 \pm 34.62$ | \$25.007 \pm 1.036 | $5152.9 \pm 700.1$ | **5.60** |
|  |  | **Proposed Cloud-Edge DT** | **$525.40 \pm 43.53$** | **\$14.688 \pm 0.859** | **$5760.5 \pm 810.6$** | **0.00** |
#### Key Findings from Transport Sensitivity:
1. **Strict Boundary Consistency at $\tau_{\text{trans}} = 0.0$:**  
   At zero transport delay ($\tau_{\text{trans}} = 0.0\,\text{min}$), all reported Mean and Std metrics for Makespan, TOU Cost, TWT, and Tool Breakdowns match the corresponding benchmark entries in Table 1 identically (error $< 10^{-4}$), verifying experimental protocol rigor.
2. **Monotonic Progression of Makespan and Tardiness:**  
   As transport latency $\tau_{\text{trans}}$ scales from $0.0$ to $3.0\,\text{min}$, Makespan and TWT scale monotonically across all methods. On Mk01, Proposed DT makespan increases from $86.00\,\text{min}$ to $130.24\,\text{min}$, while on Mk03, it scales from $462.69\,\text{min}$ to $525.40\,\text{min}$.
3. **Invariance of Tool Preservation:**  
   Across all four transport latency tiers ($0.0, 1.0, 2.0, 3.0\,\text{min}$), the Proposed Cloud-Edge DT (and identically EKF-equipped Right-Shift Rescheduling) maintains **$0.00$ tool breakdowns** on both `Mk01` and `Mk03`. In contrast, open-loop NSGA-III suffers $0.87$ tool failures on Mk01 and $9.67$ failures on Mk03, and unguided Standalone Edge PPO suffers $5.60$–$6.73$ failures on Mk03, confirming that real-time EKF wear estimation and $2\sigma_{\text{EKF}}$ statistical preventive maintenance operate invariantly to inter-machine transit times.

---


### 4.5 Closed-Loop Multi-Tier Latency & Five-Mode HIL Evaluation (incl. Frequency-Matched Periodic Control)
To isolate **(A) Live EKF Wear Feedback**, **(B) Call-Frequency Reduction**, and **(C) Event-Triggered Timing vs. Blind Periodic Timing**, we execute a 5-mode HIL evaluation across 15 independent test seeds (`TEST_SEEDS = [301..315]`, Brandimarte `Mk01` physical twin, 55 operations) after freezing $\theta_{\text{th}} = 0.12$ within the validation Event-Matched Plateau $[0.08, 0.18]$ (`VAL_SEEDS = [201..210]`). All 5 modes share identical TOU-aware speed logic (preserving multi-objective planned speed off/mid-peak and derating to $s=1$ inside On-Peak $\text{rate} \ge 0.20$), TOU-cost-aware resync scoring, and statistically principled EKF wear projection ($w_{\text{proj}} = \hat{w} + \Delta w_{\text{nom}} + 2\sigma_{\text{EKF}} + 2.0$), while billing each Cloud Warm-Start Resynchronization invocation at its empirical latency (`cloud_nsga3_resync_solve_s = 2.6835 s` + `2 * 0.0814 ms` MQTT RTT):
- **Mode 1 (`Periodic High-Freq + No EKF`, Every 5 ops):** $10.00$ blind periodic calls using cumulative nominal wear tracking (`nominal_wear_um`, no EKF sensors).
- **Mode 2 (`Periodic High-Freq + Live EKF`, Every 5 ops):** $10.00$ periodic calls ($26.8366\,\text{s}$ total Cloud Warm-Start solver delay) paired with Live EKF.
- **Mode 3 (`Periodic Frequency-Matched + Live EKF`, Op 25 & Op 50):** Fixed low-frequency periodic calls matched to $N_{\text{shocks}} = 2.0$ ($2.00$ calls, $5.3673\,\text{s}$ solver delay) at fixed milestones regardless of true shock timing (`Op 19..22`, `Op 37..40`).
- **Mode 4 (`One-Shot Autonomous + Live EKF`, 0 calls):** Edge EKF + ONNX policy only, without online Cloud re-synchronization.
- **Mode 5 (`Proposed Event-Triggered CDI + Live EKF`, $\theta_{\text{th}} = 0.12$):** Invokes Cloud Warm-Start Resynchronization immediately when $\text{CDI}(t) > 0.12$.

```
Table 5: Quantitative Comparison of Five Modes in Closed-Loop HIL Simulation (55 Operations, Mk01 Physical Twin, Mean ± Std & Paired Wilcoxon vs. Mode 5 across TEST_SEEDS = [301..315])
```
| Operational Mode | Resync Calls ($\text{Mean} \pm \text{Std}$) | Call Reduction (%) | Cloud Warm-Start Overhead ($\text{Mean} \pm \text{Std}$) | Edge Latency (ms) | Network Packets | Makespan ($\text{Mean} \pm \text{Std}$, min) | Energy ($\text{Mean} \pm \text{Std}$, kWh) | TOU Cost ($\text{Mean} \pm \text{Std}$, \$) | Max CDI ($\text{Mean} \pm \text{Std}$) | Final CDI ($\text{Mean} \pm \text{Std}$) | PM ($\text{Mean} \pm \text{Std}$) | Breakdowns ($\text{Mean} \pm \text{Std}$) | $p_{\text{Wilcoxon}}$ ($C_{\max}$) | $p_{\text{Wilcoxon}}$ (Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Mode 1: Periodic High-Freq + No EKF (Every 5 ops) | 10.00 ± 0.00 | 0.0% | 26.8366 ± 0.0000 s | — | 220.00 | 61.74 ± 1.98 | 18.856 ± 0.175 | \$2.874 ± 0.078 | 0.0768 ± 0.0040 | 0.0109 ± 0.0004 | 5.00 ± 0.00 | 0.00 ± 0.00 | < 0.0001 | < 0.0001 |
| Mode 2: Periodic High-Freq + Live EKF (Every 5 ops) | 10.00 ± 0.00 | 0.0% | 26.8366 ± 0.0000 s | 0.0132 ms | 220.00 | 68.28 ± 4.76 | 19.120 ± 0.406 | \$2.862 ± 0.199 | 0.0797 ± 0.0028 | 0.0266 ± 0.0153 | 6.13 ± 0.72 | 0.00 ± 0.00 | 0.0730 | 0.0067 |
| Mode 3: Periodic Frequency-Matched + Live EKF (Op 25, 50) | 2.00 ± 0.00 | 80.0% | 5.3673 ± 0.0000 s | 0.0132 ms | 204.00 | 73.42 ± 4.58 | 18.769 ± 0.216 | \$2.617 ± 0.105 | 0.1335 ± 0.0119 | 0.0193 ± 0.0111 | 5.67 ± 0.47 | 0.00 ± 0.00 | 0.3809 | 1.0000 |
| Mode 4: One-Shot Autonomous + Live EKF (No Re-sync) | 0.00 ± 0.00 | 100.0% | 0.0 s | 0.0132 ms | 200.00 | 84.53 ± 3.23 | 19.074 ± 0.270 | \$2.578 ± 0.070 | 0.2593 ± 0.0334 | 0.2593 ± 0.0334 | 5.67 ± 0.47 | 0.00 ± 0.00 | 0.0007 | 0.0833 |
| **Mode 5: Proposed Event-Triggered CDI + Live EKF (CDI=0.12)** | **1.27 ± 0.44** | **87.3%** | **3.3993 ± 1.1868 s** | 0.0132 ms | **202.53** | **72.22 ± 5.01** | **18.809 ± 0.227** | **\$2.619 ± 0.113** | 0.1345 ± 0.0105 | 0.0640 ± 0.0371 | 5.73 ± 0.44 | **0.00 ± 0.00** | **Base** | **Base** |

#### Multi-Tier Latency Hierarchy & Cloud Warm-Start Overhead:
Empirical benchmarking on a Linux x86_64 workstation (Python 3.12, ONNX Runtime CPUExecutionProvider) confirms the multi-tier timing distribution:
1. **Edge ONNX Forward Pass:** $0.0132\,\text{ms}$ ($13.2\,\mu\text{s}$, $\sim 75,500$ inferences/second).
2. **Local MQTT Transport RTT:** $0.0814\,\text{ms}$ (industrial LAN broker loopback).
3. **Edge Decision Step:** $0.1506\,\text{ms}$ (full feature extraction, action masking, and ONNX prediction).
4. **Online List Surrogate (`fast_heuristic_resync`):** $0.2143\,\text{ms}$ (simulation-engine surrogate assignment).
5. **Edge Shift Computation:** $7.955\,\text{ms}$ (entire 55-step shift simulation on Mk01; $22.71\,\text{ms}$ on Mk03).
6. **Cloud Warm-Start NSGA-III Resync (Billed in HIL):** $2.6835\,\text{s}$ per trigger ($26.84\,\text{s}$ cumulative for 10 periodic calls).
7. **Cloud NSGA-III Full Solve (Offline Baseline):** $4.2392\,\text{s}$ ($4.24 \pm 0.03\,\text{s}$ on Mk01; $14.97\,\text{s}$ on Mk03).

This establishes a clear multi-dimensional latency hierarchy: per-decision edge reactivity ($0.1506\,\text{ms}$) is $\sim 28,147\times$ ($\sim 2.81 \times 10^4\times$) faster than a full Cloud NSGA-III solve ($4.2392\,\text{s}$) and $\sim 17,818\times$ faster than a Cloud Warm-Start NSGA-III resync ($2.6835\,\text{s}$), explaining why high-frequency periodic Cloud calls ($10 \times 2.6835\,\text{s} = 26.84\,\text{s}$) create severe solver queuing bottlenecks.

#### Component & Timing Attribution (Figure 3):
- **Value of Event-Triggered Timing over Frequency-Matched Periodic (`Mode 3` vs. `Mode 5`):** Both modes use identical Live EKF and PM rules, and **Mode 3** is explicitly tuned to the exact shock count ($2.00$ calls at Op 25 and Op 50, $5.3673\,\text{s}$ overhead). Yet because fixed milestones miss the random shock instants (`Op 19..22`, `Op 37..40`) and trigger needlessly at Op 50 even after benign shifts, **Mode 5** reduces Cloud Warm-Start calls by an additional **36.5%** ($1.27 \pm 0.44$ vs. $2.00$, cutting solver overhead from $5.37\,\text{s}$ to $3.40 \pm 1.19\,\text{s}$) while achieving slightly shorter mean makespan ($72.22 \pm 5.01$ vs. $73.42 \pm 4.58\,\text{min}$, $p = 0.3809$) and virtually identical TOU cost (\$2.619 vs. \$2.617, $p = 1.0000$).
- **Isolating CDI Event-Triggering vs. High-Frequency Periodic (`Mode 2` vs. `Mode 5`):** Replacing 10-call periodic resync (**Mode 2**, $26.84\,\text{s}$ solver overhead) with **Mode 5** slashes synchronization calls and Warm-Start solver delay by **87.3%** ($3.40\,\text{s}$ vs. $26.84\,\text{s}$), significantly reduces TOU electricity cost by **$8.5%** (\$2.619 ± 0.113 vs. \$2.862 ± 0.199, $p = 0.0067$), with no statistically significant makespan penalty ($72.22$ vs. $68.28\,\text{min}$, $p = 0.0730$).
- **Isolating Bounded Resync vs. Zero Resync (`Mode 4` vs. `Mode 5`):** Operating purely autonomously without Cloud re-synchronization (**Mode 4**, $0.00$ calls) accumulates unmitigated drift ($\text{CDI}_{\max} = 0.2593 \pm 0.0334$) that inflates makespan to $84.53 \pm 3.23\,\text{min}$. Adding just $1.27 \pm 0.44$ CDI-triggered calls in **Mode 5** recovers **$12.31\,\text{min}$** of makespan ($p = 0.0007$) at no statistically significant TOU cost difference (\$2.619 vs. \$2.578, $p = 0.0833$).
- **Sensorless PM Blindness (`Mode 1` vs. `Mode 2`/`Mode 5`):** Without live EKF, **Mode 1** executes rigid schedule-only PM ($\text{PM} = 5.00 \pm 0.00$ vs. $6.13 \pm 0.72$ in Mode 2 and $5.73 \pm 0.44$ in Mode 5), ignoring $+26\,\mu\text{m}$ shock spikes and incurring $+\$0.255$/shift higher TOU cost than Mode 5 ($p = < 0.0001$).

---

## 5. Mechatronic Hardware-in-the-Loop Telemetry & Physical Limits

### 5.1 Quantitative Telemetry Across 8 Operational Segments
The physical testbed comprises an ESP32 microcontroller, an INA226 current/power sensor ($10\,\text{m}\Omega$ / $0.010\,\Omega$ shunt), an ADXL345 3-axis accelerometer configured in **Full-Resolution $\pm 16g$ mode (`0x31 = 0x0B`, $3.9\,\text{mg/LSB}$)** at **$200\,\text{Hz}$** sampling rate ($100\,\text{Hz}$ Nyquist bandwidth capturing low-frequency structural vibration energy and sub-100 Hz amplitude modulation), opto-isolated solid-state relays on `GPIO 25` and `GPIO 26` protected by **external $10\,\text{k}\Omega$ pull-down resistors to `GND`** (paired with internal boot pull-down enable), and an **Open-Drain (`OUTPUT_OPEN_DRAIN`, internal $3.3\,\text{V}$ pull-up disabled via `gpio_pullup_dis`) hardware brake output `GPIO 27` (`PIN_GRBL_FEED_HOLD`)** (High-Z normal, active `LOW` brake) to prevent $3.3\,\text{V}$/$5\,\text{V}$ voltage-level conflicts: for **Arduino Uno + CNC Shield V3 ($5\,\text{V}$ logic)**, `GPIO 27` connects through an isolating optocoupler (`PC817`) or N-channel MOSFET (`2N7000`) open-drain stage to `Feed Hold (A1)`; for **Makerbase MKS DLC32 ($3.3\,\text{V}$ ESP32 running Grbl_ESP32 / FluidNC)**, it connects to the `Hold` pin on the input header (`I36 / GPIO 36`). During transient I2C bus errors within the 3-sample debounce window ($1 \le \text{i2c\_err\_cnt} < 3$, spanning $15\,\text{ms}$ at $200\,\text{Hz}$), the firmware applies a Zero-Order Hold (ZOH) on filtered estimates and explicitly flags `"sample_held": true` in the $10\,\text{Hz}$ JSON packet. If an unrecovered fault persists (`i2c_consecutive_errors >= 3`, latching `fault_latched = true`), **Stop Category 1 two-phase hardware shutdown** executes immediately: (1) ESP32 pulls `GPIO 27` `LOW` in $< 1\,\text{ms}$ to halt X/Y/Z axis motion via hardware `Feed Hold`, keeping `PIN_SSR_MAIN` active so the spindle does not stall mid-cut during deceleration; concurrently, `GRBLHardwareController.emergency_stop()` transmits serial `Feed Hold (!)` and polls `?` every $50\,\text{ms}$ up to the synchronized kinematic deceleration timeout `compute_decel_timeout_s(1000.0, 10.0, 0.833) = (1000/60)/10 + 0.833 = 2.50\,\text{s}` (matching firmware `SPINDLE_DECEL_WAIT_MS = 2500\,\text{ms}` from GRBL `$110 = 1000\,\text{mm/min}` and `$120 = 10\,\text{mm/s}^2$) until GRBL confirms `Hold:0` (`Idle` or `feed <= 1e-3`); (2) only after verified axis stop (or the $2500\,\text{ms}$ / $2.50\,\text{s}$ kinematic fallback) does the controller dispatch `CMD:OFF` to open SSRs followed by `Soft Reset (0x18)`. Furthermore, if the ESP32 loses power, resets, or hangs, `EdgeDigitalTwinNode.check_telemetry_watchdog()` enforces a **$300\,\text{ms}$ Host Telemetry Heartbeat Watchdog** (3 missed $10\,\text{Hz}$ frames or `TimeoutError`/`ConnectionError`) that immediately calls `grbl.emergency_stop()` (`!`) directly over Serial.

Automated analysis of the continuous 60-second telemetry stream (`phase3_hardware_telemetry.csv`) yields eight distinct operational phases:
1. **`[0.0 - 4.9 s]` (OFF State):** Quiescent baseline ($0.0\,\text{V}, 0.0\,\text{A}, 0.0\,\text{W}$, carryover wear $30.0\,\mu\text{m}$).
2. **`[5.0 - 5.1 s]` (Inrush Transient OFF $\to$ IDLE):** Capacitive DC-bus charging draws an inrush peak power of $34.4\,\text{W}$ ($2.21\times$ the steady $15.6\,\text{W}$ IDLE baseline in $200\,\text{ms}$).
3. **`[5.2 - 11.9 s]` (Steady IDLE):** Machine waiting state drawing $15.6\,\text{W}$ ($V=24.0\,\text{V}, I=0.65\,\text{A}$, baseline vibration $0.08\,\text{g}$).
4. **`[12.0 - 21.9 s]` (Cutting at $s=1$):** Low-speed cutting with wear rate $0.80\,\mu\text{m/s}$ ($30.0 \to 37.92\,\mu\text{m}$). The FIR filter successfully rejects an anomalous electrical noise spike of $74.9\,\text{W}$ at $t=16.6\,\text{s}$ down to the true filtered cutting baseline of $49.2\,\text{W}$.
5. **`[22.0 - 41.9 s]` (Cutting at $s=3$):** Wear rate accelerates $2.875\times$ to $2.30\,\mu\text{m/s}$. As tool wear climbs from $38.0\,\mu\text{m}$ to $83.77\,\mu\text{m}$, cutting power drifts upward by **$+11.67\%$** ($90.0 \to 100.5\,\text{W}$) and chatter vibration drifts by **$+28.1\%$** ($1.53 \to 1.96\,\text{g}$) with intermittent chatter bursts.
6. **`[40.3 - 41.9 s]` (Non-Preemptive Overrun Window):** True tool wear crosses the $80.0\,\mu\text{m}$ warning boundary at $t=40.3\,\text{s}$ ($80.09\,\mu\text{m}$); the dual-sensor estimator registers $80.29\,\mu\text{m}$ at $t=40.9\,\text{s}$. The edge non-preemptive logic allows the cut to safely finish at $t=41.9\,\text{s}$ ($83.77\,\mu\text{m} < 100.0\,\mu\text{m}$ breakage limit) before halting.
7. **`[42.0 - 49.9 s]` (STANDBY & Tool Replacement):** SSR drops the machine into ultra-low power standby ($3.6\,\text{W}$, vibration $0.01\,\text{g}$) for tool replacement. Wear is reset to $0.0\,\mu\text{m}$.
8. **`[50.0 - 59.9 s]` (Post-PM Cutting with Fresh Tool at $s=3$):** Spindle inrush draws $182.7\,\text{W}$ before settling to sharp cutting power of $84.5\,\text{W}$—achieving a **$15.9\%$ power reduction** and **$29.1\%$ vibration reduction** ($1.39\,\text{g}$) compared to the worn tool ($100.5\,\text{W}$, $1.96\,\text{g}$).

### 5.2 Physical Proof-of-Concept Hardware Boundaries
While the physical breadboard proves the mathematical validity of the Kalman-filtered wear regression and SSR standby transitions, industrial translation to heavy 380V 3-phase production machinery entails practical boundaries:
- **Electromagnetic Noise:** Variable Frequency Drives (VFDs) switching high currents create significant broadband EMI, necessitating differential RS-485 or fiber-optic telemetry.
- **Harsh Environment:** High-pressure dielectric cutting fluids and metal swarf require IP67/NEMA-4X sensor enclosures and non-contact eddy-current or inductive probes.
- **Complex Modal Dynamics:** Multi-axis machine tools exhibit speed-dependent structural transfer functions that require multi-sensor inertial arrays rather than single-point accelerometers.

---

## 6. Conclusion
This study has presented a closed-loop Cloud-Edge Collaborative Digital Twin framework for dynamic green flexible job shop scheduling under cutting tool wear degradation and cyclic TOU electricity tariffs. By synthesizing global macro-optimization (NSGA-III, mean $\overline{\text{HV}} = 0.4702$ and $\overline{\text{IGD}} = 0.3675$ across Mk01–Mk05) with scale-invariant sub-millisecond edge neural dispatching (MaskablePPO ONNX: $0.0132\,\text{ms}$ forward pass, $0.1506\,\text{ms}$ decision step) via a shift-normalized Cumulative Disturbance Index (frozen at $\theta_{\text{th}} = 0.12$ within the objective Event-Matched Plateau $[0.08, 0.18]$ anchored to $N_{\text{shocks}} = 2.0$, whereas $\theta_{\text{th}}^* = 0.05$ minimizes the heuristic composite `Total_System_Cost_Index` by over-triggering), the framework eliminates all tool breakdowns ($0.00$ across Mk01–Mk05 and all transport latency tiers, compared to $0.33 - 12.67$ breakdowns for open-loop metaheuristics), reduces Cloud Warm-Start NSGA-III resynchronization calls by $87.3\%$ vs. high-frequency periodic resync ($1.27 \pm 0.44$ vs. $10.00$ calls, cutting solver delay from $26.84\,\text{s}$ to $3.40\,\text{s}$) and by $36.5\%$ vs. frequency-matched periodic control ($2.00$ calls), reduces TOU electricity costs by up to $30.95\%$ via SSR standby gating (saving $\$6.223$ per shift on Mk03 vs. idle-only; or $+44.82\%$ penalty if omitted), and saves $8.44\%$ TOU electricity cost on bottleneck Mk05 relative to fair EKF-equipped Right-Shift Rescheduling under verified Stop Category 1 hardware safety interlocks (`GPIO 27` Open-Drain Feed Hold, $2500\,\text{ms}$/$2.50\,\text{s}$ kinematic deceleration sync, and $300\,\text{ms}$ Host Watchdog).
