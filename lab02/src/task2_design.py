"""Task 2: build the parameters for W=16 and W=64 and check shapes against the design."""

import numpy as np

from common import SEED
from model import count_params, init_params

WIDTHS = [16, 64]


def expected_shapes(w: int) -> dict:
    return {"W1": (64, w), "b1": (w,), "W2": (w, w), "b2": (w,), "W3": (w, 10), "b3": (10,)}


def main() -> None:
    for w in WIDTHS:
        params = init_params(w, np.random.default_rng(SEED))
        print(f"\n=== W = {w} ===")
        for name, p in params.items():
            ok = p.shape == expected_shapes(w)[name]
            print(f"  {name}: {str(p.shape):<10} size {p.size:>5}  {'OK' if ok else 'WRONG'}")
        print(f"  total parameters: {count_params(params)}")
        W1 = params["W1"]
        print(f"  W1 mean {W1.mean():+.5f}  std {W1.std():.5f}  (target 0, 0.01)")


if __name__ == "__main__":
    main()
