"""Task 7: interrupt training after epoch 3 (hard exit, like a killed job), resume from the checkpoint, finish 10 epochs.

The reference is the Task 1 baseline: identical config, never interrupted. If resume restores everything
(weights, Adam moments, RNG and shuffle state), the two curves should match epoch for epoch.
"""

import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS_DIR, run_training

STOP = 3

if __name__ == "__main__":
    args = ["mlflow.experiment=lab06-task7-resume", "mlflow.run_name=resume"]
    first = run_training(args + [f"training.interrupt_after_epoch={STOP}"], check=False)
    assert first is None, "the first leg should have exited early"
    s = run_training(args, resume=True)
    res = pd.DataFrame(s["history"])
    ref = pd.read_csv(RESULTS_DIR / "baseline_history.csv")
    cmp = pd.DataFrame({"epoch": res.epoch, "leg": ["before crash" if e <= STOP else "after resume" for e in res.epoch],
                        "train_loss_resumed": res.train_loss, "train_loss_reference": ref.train_loss,
                        "val_acc_resumed": res.val_acc, "val_acc_reference": ref.val_acc,
                        "abs_diff_train_loss": (res.train_loss - ref.train_loss).abs(),
                        "abs_diff_val_acc": (res.val_acc - ref.val_acc).abs()})
    cmp.to_csv(RESULTS_DIR / "resume_check.csv", index=False, float_format="%.6f")
    print(cmp.to_string(index=False, float_format="%.5f"))
    print(f"resumed run test_acc {s['test_acc']:.4f}")

    fig, (l, r) = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, title in [(l, "train_loss", "training loss"), (r, "val_acc", "validation accuracy")]:
        ax.plot(ref.epoch, ref[col], "k-", lw=3, alpha=0.3, label="uninterrupted (Task 1)")
        before, after = res[res.epoch <= STOP], res[res.epoch >= STOP]
        ax.plot(before.epoch, before[col], "o-", label="run 1, killed after epoch 3")
        ax.plot(after.epoch, after[col], "s--", label="run 2, resumed from checkpoint")
        ax.axvline(STOP + 0.5, color="grey", ls=":")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "resume.png", dpi=130)
