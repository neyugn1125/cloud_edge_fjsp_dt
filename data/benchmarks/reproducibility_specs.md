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
| **Edge ONNX Engine** | Forward CPU Policy Inference | **0.0214 ms** (0.018 - 0.027 ms) | **~1.17x10^6x** |
| **Industrial MQTT** | Round-Trip Telemetry / Trigger | **0.2240 ms** | ~111,600x |
| **Cloud NSGA-III** | 100-Gen Global Re-Optimization | **25.0 s** (25000 ms) | 1.0x (Reference) |

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
- **Hypervolume Reference Point:** $\\mathbf{r}_{\\text{ref}} = [1.1, 1.1, 1.1]$ on normalized objective space $[0, 1]^3$.
- **Statistical Hypothesis Testing:** Two-tailed Wilcoxon Signed-Rank Test with significance level $\\alpha = 0.05$.
