"""Run the whole Lab 4 pipeline in order: python lab04/src/run_all.py"""

import runpy
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
TASKS = ["task2_ingest.py", "task3_validate.py", "task4_clean.py", "task5_features.py",
         "task6_label_audit.py", "task7_hash.py"]

if __name__ == "__main__":
    for task in TASKS:
        print(f"\n########## {task} ##########")
        runpy.run_path(str(SRC / task), run_name="__main__")
