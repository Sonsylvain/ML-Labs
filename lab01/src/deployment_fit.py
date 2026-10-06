"""Task 4: check each model's measurements against the Cloud/Edge/Mobile/TinyML budgets.

Memory used for the check is the peak process RSS during single-sample inference, i.e.
what a deployed Python/scikit-learn service would actually occupy. A tier "fits" when
latency, model size and memory are all within its limits. Memory ranges are checked
against the upper bound (largest device in the tier) and, separately, against the lower
bound (smallest device in the tier) to show how robust the fit is.
"""

import pandas as pd

from common import RESULTS_DIR

KB = 1 / 1024  # MB

# name: (memory lower bound MB, memory upper bound MB, latency ms, model size MB)
BUDGETS = {
    "Cloud": (1024, float("inf"), 100, 500),
    "Edge": (256, 1024, 50, 50),
    "Mobile": (64, 256, 20, 10),
    "TinyML": (0, 256 * KB, 10, 100 * KB),
}


def fmt_limit(lo: float, hi: float) -> str:
    if hi == float("inf"):
        return f">= {lo:g} MB"
    if lo == 0:
        return f"<= {hi * 1024:g} KB"
    return f"{lo:g}-{hi:g} MB"


def main() -> None:
    cost = pd.read_csv(RESULTS_DIR / "system_cost.csv")
    rows = []
    for _, m in cost.iterrows():
        size_mb = m["model_size_bytes"] / 1024**2
        for tier, (mem_lo, mem_hi, lat_max, size_max) in BUDGETS.items():
            # Cloud guarantees at least 1 GB, so its effective memory ceiling is 1 GB.
            mem_ceiling = mem_lo if mem_hi == float("inf") else mem_hi
            lat_ok = m["infer_latency_median_ms"] <= lat_max
            size_ok = size_mb <= size_max
            mem_ok = m["infer_peak_rss_mb"] <= mem_ceiling
            mem_ok_smallest = m["infer_peak_rss_mb"] <= (mem_lo or mem_hi)
            rows.append({
                "model": m["model"],
                "tier": tier,
                "memory_budget": fmt_limit(mem_lo, mem_hi),
                "latency_budget_ms": lat_max,
                "size_budget": f"{size_max:g} MB" if size_max >= 1 else f"{size_max * 1024:g} KB",
                "peak_rss_mb": round(m["infer_peak_rss_mb"], 1),
                "latency_ms": round(m["infer_latency_median_ms"], 4),
                "size_kb": round(m["model_size_kb"], 2),
                "memory_ok": mem_ok,
                "memory_ok_smallest_device": mem_ok_smallest,
                "latency_ok": lat_ok,
                "size_ok": size_ok,
                "fits": bool(mem_ok and lat_ok and size_ok),
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "deployment_fit.csv", index=False)

    tick = {True: "yes", False: "no"}
    md = df.replace(tick).astype(str)
    lines = ["| " + " | ".join(md.columns) + " |", "|" + "---|" * len(md.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in md.itertuples(index=False)]
    (RESULTS_DIR / "deployment_fit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md.to_string(index=False))


if __name__ == "__main__":
    main()
