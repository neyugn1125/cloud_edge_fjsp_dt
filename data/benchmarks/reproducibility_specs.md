# Experimental Reproducibility Specifications & Single Source of Truth

**Project:** A Cloud-Edge Collaborative Digital Twin Framework for Energy-Aware Flexible Job Shop Scheduling  
**Target Publication:** IEEE Transactions on Industrial Informatics / Elsevier Journal of Manufacturing Systems  
**Generated At:** Evaluation Protocol Engine v5.5A

---

## 1. Disjoint Random Seed Sets (Zero Data Leakage Protocol)

To eliminate any potential data snooping or circular validation, random seeds are strictly partitioned into three disjoint sets:

- **Training Seeds (`TRAIN_SEEDS`):** `[42, 101, 2026]`  
  *Purpose:* Training the Edge `MaskablePPO` policy over 66,000 timesteps across 4 parallel environments.
- **Validation Seeds (`VAL_SEEDS`):** `[201, 202, 203, 204, 205, 206, 207, 208, 209, 210]` (10 seeds)  
  *Purpose:* Tuning the Cumulative Disturbance Index synchronization threshold $\\theta_{\\text{th}}$ and hyperparameters.
- **Testing Seeds (`TEST_SEEDS`):** `[301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315]` (15 seeds)  
  *Purpose:* Independent evaluation for all reported tables and figures (Phase 2, Phase 4 HIL, Phase 5 Wilcoxon, Phase 5 Ablation).

> **Verification Assertion:** `set(TRAIN_SEEDS).isdisjoint(VAL_SEEDS) and set(VAL_SEEDS).isdisjoint(TEST_SEEDS) == True`.

---

## 2. Shift-Normalized Cumulative Disturbance Index CDI(t)

$$\text{CDI}(t) = \alpha \frac{|C_{\text{act}}(t) - C_{\text{base}}(t)|}{C_{\text{base\_total}}} + (1 - \alpha) \frac{|E_{\text{act}}(t) - E_{\\text{base}}(t)|}{E_{\text{base\_total}}}$$

- **Formula:** `CDI(t) = alpha * abs(c_actual_t - c_base_t) / c_base_total + (1.0 - alpha) * abs(e_actual_t - e_base_t) / e_base_total`

- **Normalization Basis:** Divided by total shift planned quantities ($C_{\\text{base\_total}}$, $E_{\\text{base\_total}}$) rather than instantaneous values $C_{\\text{base}}(t)$, preventing division-by-zero singularities at shift onset ($t \\to 0$).
- **Optimal Knee Point Threshold:** $\\theta_{\\text{th}} = 0.12$.
  - At $\\theta_{\\text{th}} \\le 0.12$: Autonomous Edge compensation absorbs 100% of micro-disturbances locally.
  - At $\\theta_{\\text{th}} > 0.12$: Edge publishes urgent MQTT rescheduling request (`dt/fjsp/edge/trigger_reschedule`, QoS 1) to Cloud.

---

## 3. Hardware Latency Benchmarks & Speedup

| Execution Tier | Latency Metric | Physical Measurement | Speedup vs. Cloud Solver |
| :--- | :--- | :---: | :---: |
| **Edge ONNX Engine** | Forward CPU Policy Inference | **0.0132 ms** (0.011 - 0.025 ms) | **~321,152x (~3.21x10^5x)** |
| **Edge Decision Step** | Obs + Mask + Policy Inference | **~0.1506 ms** (P95: 0.1638 ms) | **~28,149x (~2.81x10^4x)** |
| **Edge Full Shift (Mk01)** | 55 Operations Complete Shift | **~7.955 ms** (P95: 8.016 ms) | **~533x** |
| **Edge Full Shift (Mk03)** | 150 Operations Complete Shift | **~22.706 ms** (P95: 23.162 ms) | ~659x |
| **Industrial MQTT** | Round-Trip Telemetry / Trigger | **0.0814 ms** | ~52,080x |
| **Cloud NSGA-III (Resync)**| Remaining Operations Re-Solve | **~2.683 s** (P95: 2.731 s) | 1.6x vs Full Solve |
| **Cloud NSGA-III (Full)**  | 100-Gen Global Re-Optimization | **4.24 s** (4239 ms) | 1.0x (Reference) |

---

## 4. State, Action, and Reward Specifications

### 4.1 Observation Vector (87 Dimensions in $[0.0, 1.0]$)
- **Job States ($10 \\times 5 = 50$):** `[progress, rem_work_ratio, ready_time_norm, urgency_slack, priority_weight]`
- **Machine States ($6 \\times 5 = 30$):** `[m_ready_norm, tool_wear_ratio, power_drift, is_processing, is_standby]`
- **Global Factory States ($7$):** `[time_norm, tou_rate_norm, energy_norm, active_power_ratio, w_cmax, w_energy, w_tardiness]`

### 4.2 Action Space (Discrete 360)
$$\\text{Action} = j \\times (M \\times S \\times P) + m \\times (S \\times P) + s \\times P + p$$
where $j \\in [0..9]$, $m \\in [0..5]$, $s \\in [0..2]$ (Speed levels 1, 2, 3), and $p \\in [0..1]$ (PM replacement).

### 4.3 Cloud-Guided Reward Shaping
$$r_t = - \\left( w_1 \\frac{\\Delta C_{\\max}}{78.0} + w_2 \\frac{\\Delta \\text{Cost}}{\\$2.72} + w_3 \\frac{\\Delta \\text{Tardiness}}{345.0} + P_{\\text{bd}} \\right) + r_{\\text{cloud}} + r_{\\text{speed}}$$
- $r_{\\text{cloud}} = +0.15$ (Machine match) $+ 0.08$ (Sequence match).
- $r_{\\text{speed}} = +0.08$ (High-speed off-peak) $+ 0.12$ (Eco on-peak) $- 0.10$ (High-speed on-peak penalty).

---

## 5. Multi-Objective Optimization Hyperparameters

### 5.1 Cloud NSGA-III & MOEA/D
- **Population Size:** 100
- **Generations:** 100
- **Crossover:** SBX ($p_c = 0.90, \\eta_c = 15.0$)
- **Mutation:** Polynomial Mutation ($p_m = 1/n_{\\text{var}}, \\eta_m = 20.0$)
- **Reference Directions:** Das-Dennis 3 objectives, 12 partitions (91 reference vectors)

### 5.2 Edge MaskablePPO
- **Network Architecture:** Separate Actor-Critic MLP with `[256, 128]` hidden layers.
- **Learning Rate:** $3 \\times 10^{-4}$, $\\gamma = 0.99$, $\\lambda_{\\text{GAE}} = 0.95$.
- **Entropy Coefficient:** $0.015$, Batch size: $220$, Buffer size: $2200$ ($4 \\times 550$).

---

## 6. Multi-Objective Performance Indicators
- **Reference Front $PF^*$:** Non-dominated front extracted from the union of all Pareto approximations across all 15 independent test seeds.
- **Hypervolume Reference Point:** $\mathbf{r}_{\text{ref}} = [1.1, 1.1, 1.1]$ on normalized objective space $[0, 1]^3$.
- **Statistical Hypothesis Testing:** Two-tailed Wilcoxon Signed-Rank Test with Holm-Bonferroni correction ($p_{\text{holm}}$), Cliff's Delta ($\\delta$), and Vargha-Delaney effect size ($\\hat{A}_{12}$).

---

## 7. Unified Table of Experimental Regimes

To maintain absolute experimental transparency, every evaluation in this study is classified under one of three standardized regimes:

| Regime | Name & Definition | Physical Disturbance Dynamics | Benchmark Scope & Purpose | Seed Partition | Evaluation Phase |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Regime A** | Nominal Deterministic | No wear acceleration ($\\eta_{\\text{wear}} = 1.0$), nominal machining parameters, deterministic machine availability | Establishing theoretical upper bounds and initial Cloud NSGA-III / MOEA/D Pareto frontiers | Nominal seed (42) | Phase 1 & Static Cloud Optimization |
| **Regime B** | Stochastic Continuous Wear Drift | Real-world $+15\\% \\dots +35\\%$ stochastic cutting tool flank wear acceleration, dynamic power drift, random seed variation | Evaluating resilience, multi-instance scalability, and architectural component ablation under continuous physical uncertainty | 15 `TEST_SEEDS` (`[301..315]`) | Phase 2, Phase 5 Multi-Scale Wilcoxon (Mk01..Mk05), Phase 5 Ablation |
| **Regime C** | Stochastic Dual Macro-Shock HIL | Continuous wear drift coupled with 2 discrete physical macro-disturbances (Hard steel alloy batch at Op 18..22 with $+45\\% \\dots +75\\%$ wear spike & 8-min delay; Emergency tool breakdown/stoppage at Op 36..40 with 8-min delay) | Closed-loop Hardware-in-the-Loop validation and parameter sensitivity of the CDI event-triggered re-synchronization threshold $\\theta_{\\text{th}}$ | 10 `VAL_SEEDS` (`[201..210]`) | Phase 4 HIL & Phase 5 CDI Sensitivity |

---

## 8. Zero-Shot Scale Generalization Protocol

### 8.1 Architectural Principle
To eliminate the prohibitive computational overhead of training separate DRL agents for every workshop configuration, the Edge `MaskablePPO` policy is structured around a **scale-invariant observation-action mapping**:

1. **Queue Distribution Quantiles:** Instead of fixed-length job lists, the agent perceives the production backlog through 5 quantile percentiles ($10\\%, 25\\%, 50\\%, 75\\%, 90\\%$) of remaining processing time, work ratio, and due-date slack.
2. **Normalized Machine States:** Each machine's status is normalized against its own physical boundaries: flank wear ratio $w_k(t) / w_{\\text{max}}$, power drift increment $\\Delta P_{\\text{cut}} / P_{\\text{rated}}$, and normalized time until current cut completion.
3. **Cloud Prior Guidance Vector:** The Cloud macro-schedule is injected as a relative orientation tensor (suggested speed factor and machine assignment priority) rather than rigid temporal timestamps.

### 8.2 Training & Evaluation Pipeline
- **Training Base:** Trained solely on `Mk01` ($10$ jobs, $6$ machines, $55$ operations) using `TRAIN_SEEDS = [42, 101, 2026]` across 66,000 timesteps.
- **Zero-Shot Transfer:** Exported directly to ONNX Runtime (`edge_policy.onnx`) and evaluated on `Mk02` (58 ops, 6 machines), `Mk03` (150 ops, 8 machines), `Mk04` (90 ops, 8 machines), and `Mk05` (106 ops, 4 machines) on `TEST_SEEDS = [301..315]` without any fine-tuning or parameter retraining.
- **Empirical Validation:** Zero-shot deployment successfully prevents all tool breakdowns ($0.00$ BDs across all 5 benchmark instances) while dynamically throttling speeds to minimize TOU electricity costs under the cyclic tariff.

---

## 9. Academic Literature Positioning Table

Our framework is positioned within recent advances in flexible job shop scheduling, energy optimization, and industrial digital twins:

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
| **Proposed Framework** | **Closed-Loop Co-Design: Cloud NSGA-III + Edge MaskablePPO + Dual-Sensor Fusion + Shift-Normalized CDI** | **Continuous tool wear drift ($+15\\%..+35\\%$) + Stochastic dual macro-shocks (Hard alloy & Stoppage)** | **Makespan ($C_{\\max}$), TOU Electricity Cost, Total Weighted Tardiness (TWT)** | **Physics-informed Taylor dynamic wear + Dual-sensor telemetry (INA226 + ADXL345) + Pre-operation predictive PM ($0.00$ breakdowns)** | **Sub-millisecond real-time speed adaptation ($s \\in \\{1, 2, 3\\}$) synchronized with Cyclic TOU Tariff & Peak Limit ($P_{\\text{limit}} = 22.0\\,\\text{kW}$)** | **Full Co-Design: Event-triggered re-sync ($\\theta_{\\text{th}} = 0.12$) reducing Cloud calls by 80% while retaining sub-millisecond edge reactivity (0.0132 ms, 0.0214 ms legacy)** |

*Academic Positioning & Novelty Formulation:* In strict alignment with rigorous scholarly standards, we do not claim to be the "first" to utilize Digital Twins, model tool wear, or apply MaskablePPO in manufacturing scheduling. Rather, our scientific contribution is situated in direct contrast to four distinct clusters of pioneering literature:
1. **Contrasted with Tian et al. (2023, Mathematics) and Mohammad-Rafiei et al. (2024):** While they modeled cutting tool degradation and cutting speed-dependent power drift, their solutions relied on static, offline evolutionary algorithms. They lacked real-time, closed-loop speed adaptation to safeguard tools against random stochastic wear spikes and did not evaluate dynamic cyclic TOU electricity tariffs.
2. **Contrasted with Gao et al. (2024, RCIM), Wang et al. (2023, RCIM), and Zhang et al. (2022, IEEE TII):** Although they established distributed Cloud-Edge Digital Twin and DRL frameworks for dynamic scheduling, they treated machining power consumption as idealized static constants, neglecting the physical tool flank wear dynamics that induce significant active power drift ($+10.98\\%$) and catastrophic in-cut tool breakage.
3. **Contrasted with Wu et al. (2022) and Zhu et al. (2024) / Zhu (2026):** We substantially generalize the concept of recessive disturbance accumulation and incident coefficients into a shift-normalized dual-deviation Cumulative Disturbance Index $\\text{CDI}(t)$, rigorously identifying a robust operating plateau ($\\theta_{\\text{th}} \\in [0.08, 0.15]$) that minimizes total system operational cost.
4. **Contrasted with Zhao et al. (2025, ASOC), Gong et al. (2018), and Burmeister (2024):** We extend the classic MOFJSP-TOU problem from static offline scheduling to mechatronic, hardware-in-the-loop 3-state power control (Opto-isolated SSR zero-loss standby) integrated with factory-level peak demand capping ($P_{\\text{limit}} = 22.0\\,\\text{kW}$).

---

## 10. Balanced Scientific Narrative & Physical Boundary Limits

### 10.1 Honest Assessment of `Right-Shift Rescheduling`
- *A Highly Competitive Reactive Baseline:* When inheriting high-quality global Pareto baseline schedules from Cloud NSGA-III, simple Right-Shift rescheduling serves as an exceptionally formidable baseline. Under stochastic wear drift, Right-Shift incurs minimal tool breakdowns ($0.00 - 0.07$ failures), achieves performance statistically indistinguishable from the Proposed DT on small problem instances (`Mk01`, `Mk02`, $p_{\\text{holm}} \\ge 0.2190$), and even secures a higher Hypervolume on bottleneck few-machine instances (`Mk05`: $\\text{HV}=0.5716$ vs $0.4474$, $p_{\\text{holm}} = 0.0067$).
- *The Decisive Advantage of Proposed DT:* The true superiority of the Proposed DT over Right-Shift stems from active cutting speed-scaling ($s \\in \\{1, 2, 3\\}$) and opportunistic job re-routing triggered when $\\text{CDI}(t) > \\theta_{\\text{th}}$. Proposed DT dramatically slashes Total Weighted Tardiness ($-40.0\\%$ on Mk01, $-21.9\\%$ on Mk03, $-21.5\\%$ on Mk04), achieves statistically superior Inverted Generational Distance (IGD) across medium and large shops (`Mk03`, `Mk04`, `Mk05`, $p_{\\text{holm}} \\le 0.0023$), and exhibits comprehensive multi-objective dominance on large-scale multi-machine shops (`Mk03`, `Mk04`, Cliff\\'s $\\delta = +0.92 \\dots +1.00$).

### 10.2 Objective Analysis of `Standalone Edge PPO (No Cloud Prior)`
- *Local Reactivity on Dense Benchmarks:* On dense instances with tight deadlines (`Mk03–Mk05`), Standalone Edge PPO achieves lower Total Weighted Tardiness (TWT) than the Proposed DT ($2056.8\\,\\text{min}$ vs $2816.7\\,\\text{min}$ on Mk03; $864.2\\,\\text{min}$ vs $1178.0\\,\\text{min}$ on Mk04; $3501.6\\,\\text{min}$ vs $5632.1\\,\\text{min}$ on Mk05) and a shorter makespan on Mk05 ($337.53\\,\\text{min}$ vs $390.05\\,\\text{min}$). This behavior emerges from myopic greedy reflex actions prioritizing jobs with the smallest urgency slack.
- *The Penalty of Missing Global Guidance:* Lacking the global combinatorial perspective of Cloud NSGA-III (No Cloud Prior), Standalone Edge PPO accelerates machining speeds indiscriminately through expensive On-Peak tariff windows ($[25..60]\\,\\text{min}$ at $\\$0.20/\\text{kWh}$) and creates localized machine bottlenecks. Consequently, it suffers frequent tool breakdowns ($0.80 - 4.40$ failures per episode) and inflates TOU electricity costs by **$+18.6\\%$ to $+51.5\\%$** ($+22.8\\%$ on Mk03, $+51.5\\%$ on Mk04, $+35.1\\%$ on Mk05) compared to the Proposed DT.

### 10.3 Discrepancy Between Hypervolume (HV) and IGD on Benchmark `Mk05`
- *The Bottleneck Machine Dilemma:* `Mk05` poses an extreme structural bottleneck ($15$ jobs comprising $106$ operations allocated across only $4$ machines, averaging $26.5$ operations per machine).
- *Trade-off Mechanism:* In this constrained topology, the Proposed DT prioritizes absolute physical tool preservation ($0.00$ tool breakdowns) and aggressive TOU cost minimization (achieving the lowest electricity bill across all methods: $\\$11.722 \\pm 0.150$ vs $\\$12.343 \\pm 0.783$ for Right-Shift). Decelerating to $s=1$ during peak tariff hours to avoid demand penalties unavoidably extends completion times for queued jobs, resulting in higher TWT and a compressed Hypervolume along the tardiness axis.
- *Pareto Convergence:* Crucially, while boundary solutions are truncated, the non-dominated solutions generated by Proposed DT converge exceptionally close to the true reference front center, achieving the best global Inverted Generational Distance ($\\text{IGD} = 0.3826 \\pm 0.0063$, statistically beating Right-Shift at $p_{\\text{holm}} = 0.0004$).

### 10.4 Small-Scale vs. Large-Scale Trade-offs
- **Small Problem Regimes (Mk01, Mk02):** Open-loop evolutionary algorithms (`MOEA-D`, `NSGA-III`) achieve competitive makespans (~$66 - 83$ min) on small problem sizes because they omit preventive maintenance interventions. However, under physical tool wear drift, this incurs tool breakdowns ($0.33 - 0.87$ breakdowns per episode).
- **Large Industrial Regimes (Mk03, Mk04):** On dense problems with long machining horizons (e.g. Mk03 with 150 operations spanning ~400 minutes), open-loop methods collapse completely, suffering $9.67 - 12.00$ tool breakdowns. Tool breakage forces unscheduled emergency stops, workpiece scrapping, and schedule disruption. The Proposed Cloud-Edge DT completely eliminates breakdowns ($0.00$), leading to statistically significant dominance in makespan, TOU cost, and multi-objective Pareto metrics ($p_{\\text{holm}} < 0.001$, Cliff\\'s $\\delta = +1.00$).

### 10.5 The "Blind Closed-Loop Hazard" (Scale-Dependent Behavior of Variant 1)
- On `Mk01` ($55$ operations): Variant 1 ($92.85\\,\\text{min}$, $1.80$ breakdowns) performs worse than static open-loop NSGA-III ($83.64\\,\\text{min}$, $0.87$ breakdowns). The sensor-blind AI agent misjudges wear progression, cancels scheduled PM, and accelerates to $s=3$, demonstrating the acute danger of the Blind Closed-Loop Hazard.
- On `Mk03` ($150$ operations): Static open-loop NSGA-III collapses to $514.37\\,\\text{min}$ ($9.67$ breakdowns). Variant 1, while still suffering $9.00$ tool breakdowns due to sensor blindness, leverages closed-loop re-dispatching when precedence lags accumulate, successfully holding makespan at $441.80\\,\\text{min}$ ($< 514.37\\,\\text{min}$).

### 10.6 The Nature of Variant 2 as an Operational Trade-Off
- Locking cutting speed to $s=2$ is an operational design choice rather than an unmitigated defect: it yields modest gains in makespan ($-5.39\\%$ on Mk01, $-4.48\\%$ on Mk03) and TWT ($-14.42\\%$ and $-11.44\\%$) because machines never drop to conservative speed $s=1$.
- However, it loses the agile capability to throttle cutting during high-tariff windows, increasing TOU electricity costs by $+4.71\\%$ and $+5.01\\%$.

### 10.7 Physical Proof-of-Concept Hardware Limits
- The physical testbed operates as a hardware-in-the-loop laboratory proof-of-concept using an ESP32 embedded node, INA226 power monitor, ADXL345 accelerometer, and a 24V DC CNC spindle proxy.
- Real-world deployment on heavy 380V 3-phase industrial machine tools will introduce additional challenges, including coolant ingress, high-frequency electromagnetic noise from variable frequency drives (VFDs), and complex modal vibrations requiring industrial-grade enclosure and fiber-optic communication channels.
