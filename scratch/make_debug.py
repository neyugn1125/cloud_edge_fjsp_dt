import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path.cwd()))
from scratch.test_clean_hil import clean_run_closed_loop_hil_simulation

# Let's inspect step 38 detailed numbers
with open("scratch/test_clean_hil.py") as f:
    code = f.read()

# Add debug print in step 38
code_debug = code.replace(
    'cdi_trace.append(float(cdi))',
    '''if step_i in [37, 38, 39, 40]:
            print(f"DEBUG Step {step_i}: act_m_id={act_m_id}, c_act_k={c_act_k:.2f}, c_anc={c_act_anchor:.2f}, c_plan_k={c_plan_k:.2f}, c_plan_anc={c_plan_anc:.2f}, delta_c={delta_c:.4f}, delta_e={delta_e:.4f}, cdi={cdi:.4f}")
        cdi_trace.append(float(cdi))'''
)

with open("scratch/debug_step38.py", "w") as f:
    f.write(code_debug)
