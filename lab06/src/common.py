"""Shared setup for Lab 6: config loading, seeds, the Lab 3 CNN, data loaders, evaluation and a run helper."""

import json
import random
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, random_split
from torchvision import datasets, transforms

LAB_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = LAB_DIR.parent
SRC_DIR = LAB_DIR / "src"
RESULTS_DIR = LAB_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = LAB_DIR / "config.yaml"


# ---------------------------------------------------------------- config

def _parse_value(text: str):
    """'0.1' -> 0.1, 'null' -> None, 'cosine' -> 'cosine' (YAML scalar rules)."""
    return yaml.safe_load(text)


def set_key(cfg: dict, dotted: str, value) -> None:
    *parents, leaf = dotted.split(".")
    node = cfg
    for key in parents:
        node = node[key]
    if leaf not in node:
        raise KeyError(f"unknown config key: {dotted}")
    node[leaf] = value


def load_config(path: Path = CONFIG_PATH, overrides: list[str] | None = None) -> dict:
    """Read the YAML file and apply 'a.b=value' overrides. Unknown keys are an error, not a silent no-op."""
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        key, _, value = item.partition("=")
        set_key(cfg, key, _parse_value(value))
    return cfg


def flatten(cfg: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        else:
            out[name] = value
    return out


def repo_path(relative: str) -> Path:
    return (REPO_DIR / relative).resolve()


# ---------------------------------------------------------------- seeds

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- model (unchanged from Lab 3)

class CNN(nn.Module):
    """Conv(1,16,3) -> ReLU -> MaxPool2 -> Conv(16,32,3) -> ReLU -> MaxPool2 -> Flatten -> Linear(32*7*7, 10)."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28x28 -> 14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_model(cfg: dict) -> nn.Module:
    if cfg["model"]["name"] != "CNN":
        raise ValueError(f"unknown model {cfg['model']['name']}")
    return CNN()


# ---------------------------------------------------------------- data

def load_datasets(cfg: dict):
    """Official 60k train split 55k/5k with a fixed generator (same split as Lab 3), official 10k test."""
    d = cfg["data"]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((d["mean"],), (d["std"],))])
    root = repo_path(d["root"])
    full = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
    train, val = random_split(full, [d["n_train"], d["n_val"]], generator=torch.Generator().manual_seed(cfg["seed"]))
    return train, val, test


def build_loaders(cfg: dict):
    """Returns (train, val, test, shuffle_generator).

    The shuffle order comes from its own generator, which is saved in every checkpoint, so a resumed run
    sees exactly the batches the uninterrupted run would have seen. Only the training loader uses worker
    processes: on Windows each worker costs ~10 s to start (Task 3), which the small val/test passes never repay.
    """
    train, val, test = load_datasets(cfg)
    workers = cfg["data"]["num_workers"]
    bs = cfg["training"]["batch_size"]
    extra = {"num_workers": workers}
    if workers > 0:
        extra["persistent_workers"] = cfg["data"]["persistent_workers"]
    shuffle_gen = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(train, batch_size=bs, sampler=RandomSampler(train, generator=shuffle_gen), **extra)
    val_loader = DataLoader(val, batch_size=512)
    test_loader = DataLoader(test, batch_size=512)
    return train_loader, val_loader, test_loader, shuffle_gen


# ---------------------------------------------------------------- optimisation

def build_optimizer(cfg: dict, params):
    o = cfg["optimizer"]
    if o["name"] == "adam":
        return torch.optim.Adam(params, lr=o["lr"], weight_decay=o["weight_decay"])
    if o["name"] == "sgd":
        return torch.optim.SGD(params, lr=o["lr"], momentum=o["momentum"], weight_decay=o["weight_decay"])
    raise ValueError(f"unknown optimizer {o['name']}")


def build_scheduler(cfg: dict, opt):
    s, epochs = cfg["scheduler"], cfg["training"]["epochs"]
    if s["name"] == "none":
        return None
    if s["name"] == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=s["step_size"], gamma=s["gamma"])
    if s["name"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if s["name"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=s["factor"], patience=s["patience"])
    raise ValueError(f"unknown scheduler {s['name']}")


@torch.no_grad()
def evaluate(model, loader) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        logits = model(x)
        loss += loss_fn(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    return loss / n, correct / n


# ---------------------------------------------------------------- memory

class PeakRSS:
    """Samples this process's resident set size every few ms in a background thread and keeps the maximum."""

    def __init__(self, interval_s: float = 0.005):
        self.proc = psutil.Process()
        self.interval = interval_s
        self.baseline = self.peak = self.proc.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.baseline = self.peak = self.proc.memory_info().rss
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


# ---------------------------------------------------------------- running train.py as a separate process

def run_training(overrides: list[str], resume: bool = False, check: bool = True) -> dict | None:
    """Run train.py in a fresh process (clean memory and timing per run) and return its summary.

    Returns None when the run exits early on purpose (Task 7's simulated crash) and check is False.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "summary.json"
        cmd = [sys.executable, "-W", "ignore", str(SRC_DIR / "train.py"), "--summary-out", str(out)]
        for item in overrides:
            cmd += ["--set", item]
        if resume:
            cmd.append("--resume")
        print("$ train.py " + " ".join(cmd[6:]), flush=True)
        proc = subprocess.run(cmd, cwd=REPO_DIR)
        if proc.returncode != 0:
            if check:
                raise RuntimeError(f"train.py exited with {proc.returncode}")
            return None
        return json.loads(out.read_text())


