"""Run every Lab 5 task in order: python lab05/src/run_all.py"""

import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
TASKS = ["task3_graph.py", "task4_verify.py", "task5_moons.py", "task6_framework.py"]

if __name__ == "__main__":
    for task in TASKS:
        print(f"\n########## {task} ##########")
        runpy.run_path(str(SRC / task), run_name="__main__")
