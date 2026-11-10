"""Task 5: StepLR vs CosineAnnealingLR vs ReduceLROnPlateau, each with SGD+momentum and with Adam, 10 epochs each.

Equal budget = same 10 epochs, same batch 128, same data order and initial weights. Base learning rates are the
usual defaults for each optimizer (SGD 0.02 with momentum 0.9, Adam 1e-3). Adam with a constant LR is the Task 1
baseline run, used as the no-schedule reference.
"""

import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS_DIR, run_training

EPOCHS = 10
OPTIMIZERS = {"sgd": ["optimizer.name=sgd", "optimizer.lr=0.02", "optimizer.momentum=0.9"],
              "adam": ["optimizer.name=adam", "optimizer.lr=0.001"]}
SCHEDULERS = ["step", "cosine", "plateau"]

if __name__ == "__main__":
    rows, curves = [], []
    for opt_name, opt_args in OPTIMIZERS.items():
        for sched in SCHEDULERS:
            name = f"{opt_name}-{sched}"
            s = run_training(opt_args + [f"scheduler.name={sched}", f"training.epochs={EPOCHS}",
                                         "mlflow.experiment=lab06-task5-schedules", f"mlflow.run_name={name}"])
            h = pd.DataFrame(s["history"])
            curves.append(h.assign(optimizer=opt_name, scheduler=sched))
            best = h.loc[h.val_acc.idxmax()]
            rows.append({"optimizer": opt_name, "scheduler": sched, "final_lr": h.lr.iloc[-1],
                         "final_train_loss": h.train_loss.iloc[-1], "final_val_loss": h.val_loss.iloc[-1],
                         "final_val_acc": h.val_acc.iloc[-1], "best_val_acc": best.val_acc,
                         "best_epoch": int(best.epoch), "test_acc": s["test_acc"],
                         "train_seconds": h.epoch_seconds.sum()})
    base_path = RESULTS_DIR / "baseline_history.csv"
    if base_path.exists():  # Adam, constant LR, same 10 epochs: the reference without a schedule
        curves.append(pd.read_csv(base_path).assign(optimizer="adam", scheduler="constant"))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "schedule_study.csv", index=False, float_format="%.5f")
    allc = pd.concat(curves)
    allc.to_csv(RESULTS_DIR / "schedule_study_curves.csv", index=False, float_format="%.6f")
    print(df.to_string(index=False, float_format="%.4f"))

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for col, opt_name in enumerate(OPTIMIZERS):
        sub = allc[allc.optimizer == opt_name]
        for sched, g in sub.groupby("scheduler"):
            style = "k:" if sched == "constant" else "-o"
            axes[0, col].plot(g.epoch, g.val_acc, style, ms=3, label=sched)
            axes[1, col].plot(g.epoch, g.lr, style, ms=3, label=sched)
        axes[0, col].set_title(f"{opt_name}: validation accuracy")
        axes[1, col].set_title(f"{opt_name}: learning rate used in epoch")
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel("epoch")
        axes[0, col].legend()
    for (opt_name, sched), g in allc.groupby(["optimizer", "scheduler"]):
        axes[0, 2].plot(g.epoch, g.train_loss, ("--" if opt_name == "sgd" else "-"), label=f"{opt_name}-{sched}")
    axes[0, 2].set_title("training loss (dashed = SGD)")
    axes[0, 2].legend(fontsize=7)
    axes[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "schedule_study.png", dpi=130)
