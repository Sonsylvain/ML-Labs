"""Task 5: check analytic gradients against central-difference slopes on 10 random weights.

The check runs on one training sample. With a batch, the mean in the loss divides every
gradient by B, pushing many of them down toward the ~1e-11 float64 rounding floor of the
numerical slope (loss ~2.3 stored to ~4e-16, divided by 2h = 2e-5). The relative error then
measures rounding noise, not the backprop formula. See the batch-size table printed at the end.
"""

import numpy as np

from common import SEED, load_split
from model import backward, cross_entropy, forward, init_params

WIDTHS = [16, 64]
H = 1e-5
N_CHECKS = 10
THRESHOLD = 1e-6
MIN_PASS = 9
N_TRIALS = 20  # repeat the whole check on different samples to show it is not one lucky draw
WEIGHTS = ["W1", "W2", "W3"]


def rel_error(a: float, n: float) -> float:
    return abs(a - n) / max(abs(a) + abs(n), 1e-300)


def numeric_slope(params, X, Y, name, idx) -> float:
    original = params[name][idx]
    params[name][idx] = original + H
    loss_plus = cross_entropy(forward(params, X)[0], Y)
    params[name][idx] = original - H
    loss_minus = cross_entropy(forward(params, X)[0], Y)
    params[name][idx] = original  # restore
    return (loss_plus - loss_minus) / (2 * H)


def grad_check(params, X, Y, rng) -> list[dict]:
    """10 distinct random weights (cycling W1, W2, W3) whose analytic gradient is non-zero."""
    grads = backward(params, forward(params, X)[1], Y)
    # zero gradient = ReLU-blocked path; 0 == 0 would test nothing
    pools = {name: rng.permutation(np.argwhere(grads[name] != 0)) for name in WEIGHTS}
    rows = []
    for i in range(N_CHECKS):
        name = WEIGHTS[i % 3]
        idx = tuple(pools[name][i // 3])  # walk a shuffled pool -> no weight is drawn twice
        a, n = grads[name][idx], numeric_slope(params, X, Y, name, idx)
        rows.append({"param": f"{name}{[int(j) for j in idx]}", "analytic": a, "numeric": n,
                     "rel_err": rel_error(a, n)})
    return rows


def n_passed(rows) -> int:
    return sum(r["rel_err"] < THRESHOLD for r in rows)


def main() -> None:
    X_train, _, Y_train, *_ = load_split()

    for w in WIDTHS:
        params = init_params(w, np.random.default_rng(SEED))
        rows = grad_check(params, X_train[:1], Y_train[:1], np.random.default_rng(SEED))
        print(f"\n=== W = {w}  (sample 0, h = {H}) ===")
        print(f"  {'param':<12} {'analytic':>13} {'numeric':>13} {'rel err':>10}")
        for r in rows:
            mark = "OK" if r["rel_err"] < THRESHOLD else "FAIL"
            print(f"  {r['param']:<12} {r['analytic']:>13.5e} {r['numeric']:>13.5e} {r['rel_err']:>10.2e}  {mark}")
        passed = n_passed(rows)
        print(f"  passed {passed}/{N_CHECKS} -> {'PASS' if passed >= MIN_PASS else 'FAIL'} (need >= {MIN_PASS})")

        trial_passes = [n_passed(grad_check(params, X_train[t:t + 1], Y_train[t:t + 1],
                                            np.random.default_rng(t))) for t in range(N_TRIALS)]
        ok = sum(p >= MIN_PASS for p in trial_passes)
        print(f"  robustness: {ok}/{N_TRIALS} different samples pass; checks passed per trial = {trial_passes}")

    print("\nwhy one sample: share of checks under 1e-6 vs batch size (W=16, 100 checks each)")
    params = init_params(16, np.random.default_rng(SEED))
    for B in [1, 8, 32]:
        rows = []
        for t in range(10):
            s = slice(t * B, (t + 1) * B)
            rows += grad_check(params, X_train[s], Y_train[s], np.random.default_rng(t))
        med = np.median([abs(r["analytic"]) for r in rows])
        gap = np.median([abs(r["analytic"] - r["numeric"]) for r in rows])
        print(f"  B={B:<3} pass {n_passed(rows)}/{len(rows)}   median |grad| {med:.1e}   "
              f"median |analytic - numeric| {gap:.1e}")


if __name__ == "__main__":
    main()
