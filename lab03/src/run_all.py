"""Run every Lab 3 task in order."""

import runpy
from pathlib import Path

SRC = Path(__file__).resolve().parent
TASKS = ["task1_dataset.py", "task2_models.py", "task3_flops.py", "task3_latency.py",
         "task4_train.py", "task5_plots.py"]

if __name__ == "__main__":
    for task in TASKS:
        print(f"\n########## {task} ##########")
        runpy.run_path(str(SRC / task), run_name="__main__")
