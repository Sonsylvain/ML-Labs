"""Task 4: ((a*b + c)**2).relu().sum() -- scalar engine vs PyTorch autograd vs central differences."""

import numpy as np
import pandas as pd
import torch

from common import RESULTS_DIR, SEED
from value import Value, vsum

SHAPE = (3, 4)
EPS = 1e-6
TOL = 1e-6


def f_numpy(a, b, c) -> float:
    return float(np.maximum((a * b + c) ** 2, 0).sum())


def engine_grads(a, b, c):
    A, B, C = ([Value(v) for v in arr.ravel()] for arr in (a, b, c))
    out = vsum([((x * y + z) ** 2).relu() for x, y, z in zip(A, B, C)])
    out.backward()
    return out.data, [np.array([v.grad for v in vals]).reshape(SHAPE) for vals in (A, B, C)]


def torch_grads(a, b, c):
    ta, tb, tc = (torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (a, b, c))
    out = ((ta * tb + tc) ** 2).relu().sum()
    out.backward()
    return out.item(), [t.grad.numpy() for t in (ta, tb, tc)]


def numeric_grads(a, b, c):
    grads = []
    for i in range(3):
        g = np.zeros(SHAPE)
        for idx in np.ndindex(SHAPE):
            plus, minus = [x.copy() for x in (a, b, c)], [x.copy() for x in (a, b, c)]
            plus[i][idx] += EPS
            minus[i][idx] -= EPS
            g[idx] = (f_numpy(*plus) - f_numpy(*minus)) / (2 * EPS)
        grads.append(g)
    return grads


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    a, b, c = (rng.normal(size=SHAPE) for _ in range(3))

    v_eng, g_eng = engine_grads(a, b, c)
    v_pt, g_pt = torch_grads(a, b, c)
    g_num = numeric_grads(a, b, c)

    rows = []
    for name, ge, gp, gn in zip("abc", g_eng, g_pt, g_num):
        for idx in np.ndindex(SHAPE):
            rows.append({"input": name, "index": str(idx), "engine": ge[idx], "pytorch": gp[idx],
                         "numeric": gn[idx], "abs_diff_pytorch": abs(ge[idx] - gp[idx]),
                         "abs_diff_numeric": abs(ge[idx] - gn[idx])})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "gradient_check.csv", index=False, float_format="%.12e")

    max_pt, max_num = df.abs_diff_pytorch.max(), df.abs_diff_numeric.max()
    passed = max_pt < TOL and max_num < TOL
    lines = [f"expression ((a*b + c)**2).relu().sum(), a, b, c ~ N(0,1) of shape {SHAPE}, float64, seed {SEED}",
             f"forward value: engine {v_eng:.12f}   pytorch {v_pt:.12f}   diff {abs(v_eng - v_pt):.1e}",
             f"{len(df)} gradients compared (a, b, c x {SHAPE[0] * SHAPE[1]})",
             f"max |engine - pytorch| = {max_pt:.3e}",
             f"max |engine - numeric| = {max_num:.3e}   (central difference, eps = {EPS})",
             f"tolerance {TOL}: {'PASS' if passed else 'FAIL'}"]
    text = "\n".join(lines)
    print(text)
    (RESULTS_DIR / "gradient_check.txt").write_text(text + "\n")
    assert passed
