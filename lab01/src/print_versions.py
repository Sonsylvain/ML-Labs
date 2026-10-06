"""Task 1: Printing Python and the versions of the packages pinned in requirements.txt."""

import platform
import sys
from importlib import metadata
from pathlib import Path

from packaging.version import Version

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
REQUIREMENTS = Path(__file__).resolve().parents[2] / "requirements.txt"


def pinned_requirements() -> dict:
    pins = {}
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip().lower()] = version.strip()
    return pins


def main() -> None:
    lines = [
        f"Python      : {sys.version.split()[0]} ({platform.python_implementation()})",
        f"Executable  : {sys.executable}",
        f"Platform    : {platform.platform()}",
        f"Machine     : {platform.machine()} / {platform.processor()}",
        "",
        "Pinned packages (requirements.txt):",
    ]

    pins = pinned_requirements()
    mismatches = 0
    for name, wanted in pins.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = "NOT INSTALLED"
        # Compare as PEP 440 versions: "0.1.1-2209072238" == "0.1.1.post2209072238".
        same = installed != "NOT INSTALLED" and Version(installed) == Version(wanted)
        status = "OK" if same else f"MISMATCH (want {wanted})"
        mismatches += status != "OK"
        lines.append(f"  {name:<20} {installed:<28} {status}")

    lines += ["", f"packages matching requirements.txt: {len(pins) - mismatches}/{len(pins)}"]

    text = "\n".join(lines)
    print(text)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "versions.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
