import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from scratch.test_clean_hil import clean_run_closed_loop_hil_simulation

# Let's test with job_ready_time delayed as well
# In scratch/test_clean_hil.py, let's see how CDI responds
