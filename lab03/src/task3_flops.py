"""Task 3a: torchinfo parameter counts, hand FLOPs (Ch. 4 formulas) vs thop.profile, per layer and per model.

FLOPs here = 2 x multiply-adds (one multiply + one add). thop reports MACs, so thop FLOPs = 2 x thop MACs.
"""

import io
from contextlib import redirect_stdout

import pandas as pd
import thop
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from torchinfo import summary

from common import RESULTS_DIR, set_seeds
from models import build_models

SEQ, D_MODEL, D_FF = 16, 64, 128


# ---- Ch. 4 formulas (batch 1) -------------------------------------------------------------
def linear_flops(n_in: int, n_out: int, tokens: int = 1) -> int:
    return 2 * tokens * n_in * n_out


def conv_flops(k: int, c_in: int, c_out: int, h_out: int, w_out: int) -> int:
    return 2 * k * k * c_in * c_out * h_out * w_out


def encoder_layer_flops(seq: int = SEQ, d: int = D_MODEL, ff: int = D_FF) -> dict[str, int]:
    return {
        "QKV projections": 2 * seq * d * 3 * d,
        "scores Q.K^T": 2 * seq * seq * d,
        "weights x V": 2 * seq * seq * d,
        "output projection": 2 * seq * d * d,
        "FFN (2 linears)": 2 * seq * d * ff * 2,
    }


def hand_model_flops() -> dict[str, int]:
    return {
        "MLP": linear_flops(784, 256) + linear_flops(256, 128) + linear_flops(128, 10),
        "CNN": conv_flops(3, 1, 16, 28, 28) + conv_flops(3, 16, 32, 14, 14) + linear_flops(32 * 7 * 7, 10),
        "Attention": linear_flops(49, D_MODEL, SEQ) + 2 * sum(encoder_layer_flops().values()) + linear_flops(D_MODEL, 10),
    }


# ---- tools --------------------------------------------------------------------------------
def thop_macs(module: nn.Module, x: torch.Tensor) -> float:
    with torch.no_grad():
        macs, _ = thop.profile(module, (x,), verbose=False)  # note: profile() calls module.eval()
    return float(macs)


def flop_counter(module: nn.Module, x: torch.Tensor) -> int:
    module.eval()
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        module(x)
    return counter.get_total_flops()


def measure(module, x) -> dict:
    """thop and FlopCounterMode, with the fused Transformer fast path on (default) and off."""
    out = {}
    for fastpath in (True, False):
        torch.backends.mha.set_fastpath_enabled(fastpath)
        tag = "fastpath" if fastpath else "no_fastpath"
        out[f"thop_FLOPs_{tag}"] = 2 * thop_macs(module, x)
        out[f"flopcounter_{tag}"] = flop_counter(module, x)
    torch.backends.mha.set_fastpath_enabled(True)
    return out


def main() -> None:
    set_seeds()
    models = build_models()
    x = torch.randn(1, 1, 28, 28)

    # 1) torchinfo
    buf = io.StringIO()
    params = {}
    for name, model in models.items():
        info = summary(model, input_size=(1, 1, 28, 28), verbose=0,
                       col_names=("output_size", "num_params", "mult_adds"), depth=4)
        params[name] = info.total_params
        buf.write(f"=== {name} ===\n{info}\n\n")
    (RESULTS_DIR / "torchinfo_summary.txt").write_text(buf.getvalue(), encoding="utf-8")

    # 2) one representative layer of each type
    enc = models["Attention"].encoder.layers[0]
    layers = [
        ("MLP Linear 784->256", models["MLP"].net[1], torch.randn(1, 784), linear_flops(784, 256)),
        ("CNN Conv2d 16->32 (14x14 out)", models["CNN"].net[3], torch.randn(1, 16, 14, 14),
         conv_flops(3, 16, 32, 14, 14)),
        ("Attention encoder layer 1", enc, torch.randn(1, SEQ, D_MODEL), sum(encoder_layer_flops().values())),
    ]
    layer_rows = []
    for name, module, inp, hand in layers:
        row = {"layer": name, "hand_FLOPs": hand, **measure(module, inp)}
        layer_rows.append(row)
    layer_df = pd.DataFrame(layer_rows)

    # 3) whole models
    hand = hand_model_flops()
    model_rows = []
    for name, model in models.items():
        with torch.no_grad():
            _, thop_params = thop.profile(model, (x,), verbose=False)
        model_rows.append({"model": name, "torchinfo_params": params[name], "thop_params": int(thop_params),
                           "hand_FLOPs": hand[name], **measure(model, x)})
    model_df = pd.DataFrame(model_rows)

    enc_df = pd.DataFrame([{"part": k, "hand_FLOPs": v} for k, v in encoder_layer_flops().items()])
    pd.set_option("display.width", 200)
    print("Encoder layer breakdown (hand):\n", enc_df.to_string(index=False), "\n")
    print("One layer of each type:\n", layer_df.to_string(index=False), "\n")
    print("Whole models (batch 1):\n", model_df.to_string(index=False))
    enc_df.to_csv(RESULTS_DIR / "flops_encoder_breakdown.csv", index=False)
    layer_df.to_csv(RESULTS_DIR / "flops_layer.csv", index=False)
    model_df.to_csv(RESULTS_DIR / "flops_model.csv", index=False)


if __name__ == "__main__":
    main()
