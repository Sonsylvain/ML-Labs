"""Run every Lab 6 task in order: python lab06/src/run_all.py  (about 1.5 h on a 4-thread CPU)"""

import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
TASKS = ["task1_baseline.py", "task2_profile.py", "task3_workers.py", "task4_batch.py",
         "task5_schedules.py", "task6_accumulation.py", "task7_resume.py", "export_mlflow.py"]

if __name__ == "__main__":
    for task in TASKS:
        print(f"\n########## {task} ##########", flush=True)
        runpy.run_path(str(SRC / task), run_name="__main__")
