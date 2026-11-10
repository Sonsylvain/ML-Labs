"""Task 4: batch size in [32,64,128,256,512], 5 epochs each: throughput, peak memory, final accuracy.

Learning rate is held at the config value (Adam 1e-3) for every batch size, so larger batches also mean
fewer optimizer steps in the same 5 epochs. Each run is a separate process, so peak RSS is not polluted
by the previous run.
"""

import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS_DIR, run_training

BATCHES = [32, 64, 128, 256, 512]
EPOCHS = 5

if __name__ == "__main__":
    rows, curves = [], []
    for bs in BATCHES:
        s = run_training([f"training.batch_size={bs}", f"training.epochs={EPOCHS}",
                          "mlflow.experiment=lab06-task4-batch", f"mlflow.run_name=batch{bs}"])
        h = pd.DataFrame(s["history"])
        curves.append(h.assign(batch_size=bs))
        rows.append({"batch_size": bs, "optimizer_steps": int(h.optimizer_steps.sum()),
                     "throughput_samples_s": h.throughput.mean(), "train_seconds": h.epoch_seconds.sum(),
                     "peak_rss_mb": s["peak_rss_mb"], "peak_rss_increase_mb": s["peak_rss_increase_mb"],
                     "final_train_loss": h.train_loss.iloc[-1], "final_val_acc": h.val_acc.iloc[-1],
                     "test_acc": s["test_acc"]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "batch_study.csv", index=False, float_format="%.4f")
    pd.concat(curves).to_csv(RESULTS_DIR / "batch_study_curves.csv", index=False, float_format="%.5f")
    print(df.to_string(index=False, float_format="%.4f"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, col, label in [(axes[0], "throughput_samples_s", "training throughput (samples/s)"),
                           (axes[1], "peak_rss_increase_mb", "peak RSS increase during training (MB)"),
                           (axes[2], "test_acc", f"test accuracy after {EPOCHS} epochs")]:
        ax.plot(df.batch_size, df[col], "o-")
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCHES, [str(b) for b in BATCHES])
        ax.set_xlabel("batch size")
        ax.set_title(label)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "batch_study.png", dpi=130)
