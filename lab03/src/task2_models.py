"""Task 2: build the three models, check output shapes and parameter counts."""

import pandas as pd
import torch

from common import RESULTS_DIR, set_seeds
from models import build_models


def main() -> None:
    set_seeds()
    x = torch.randn(2, 1, 28, 28)
    rows = []
    for name, model in build_models().items():
        out = model(x)
        n = sum(p.numel() for p in model.parameters())
        rows.append({"model": name, "output_shape": tuple(out.shape), "params": n,
                     "in_100k_150k_target": 100_000 <= n <= 150_000})
        print(f"{name}\n{model}\n")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(RESULTS_DIR / "model_params.csv", index=False)


if __name__ == "__main__":
    main()
