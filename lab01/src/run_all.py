"""Run every Lab 1 step in order: versions -> accuracy -> system cost -> deployment fit."""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = ["print_versions.py", "train_baselines.py", "measure_system_cost.py", "deployment_fit.py"]

for step in STEPS:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, str(SRC / step)], check=True, cwd=SRC)
