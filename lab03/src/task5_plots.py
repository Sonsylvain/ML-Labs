"""Task 5: test accuracy against parameter count and against inference latency (batch 1 and batch 128)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS_DIR

COLORS = {"MLP": "#3b6fd8", "CNN": "#d8763b", "Attention": "#2a9d6a"}


def main() -> None:
    acc = pd.read_csv(RESULTS_DIR / "test_accuracy.csv").set_index("model")
    params = pd.read_csv(RESULTS_DIR / "flops_model.csv").set_index("model")
    lat = pd.read_csv(RESULTS_DIR / "latency.csv")
    lat1 = lat[lat.batch == 1].set_index("model")
    lat128 = lat[lat.batch == 128].set_index("model")

    panels = [
        (params["torchinfo_params"], "parameters (torchinfo)", "Accuracy vs parameters"),
        (lat1["median_ms"], "latency, batch 1 (median of 100 runs, ms)", "Accuracy vs latency, batch 1"),
        (lat128["ms_per_sample"], "latency per sample, batch 128 (median of 20 runs, ms)",
         "Accuracy vs latency, batch 128"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (xs, xlabel, title) in zip(axes, panels):
        for model in acc.index:
            ax.scatter(xs[model], acc.loc[model, "test_acc"] * 100, s=90, color=COLORS[model], zorder=3)
            ax.annotate(model, (xs[model], acc.loc[model, "test_acc"] * 100), textcoords="offset points",
                        xytext=(8, 6), fontsize=9)
        ax.set_xscale("log")
        ax.margins(x=0.2, y=0.15)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("test accuracy (%)")
    fig.suptitle("Lab 3: Fashion-MNIST, 5 epochs Adam lr 1e-3 (CPU)")
    fig.tight_layout()
    out = RESULTS_DIR / "tradeoffs.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
