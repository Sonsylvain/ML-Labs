# Lab 2 — Task 2: Network design

Interactive diagram: open [network_design.html](network_design.html) in a browser (toggle W = 16 / 64, hover a weight bundle).

## Architecture

```
X (B,64) --[W1,b1]--> Z1 (B,W) --ReLU--> A1 (B,W)
         --[W2,b2]--> Z2 (B,W) --ReLU--> A2 (B,W)
         --[W3,b3]--> Z3 (B,10) --softmax--> P (B,10)
```

`B` is the batch size. Each row of `X` is one flattened 8x8 image (64 pixels in [0, 1]).

## Equations (row-vector convention)

| Layer | Equation | Shape check |
|---|---|---|
| 1 | `Z1 = X @ W1 + b1`, `A1 = max(0, Z1)` | (B,64)·(64,W) = (B,W) |
| 2 | `Z2 = A1 @ W2 + b2`, `A2 = max(0, Z2)` | (B,W)·(W,W) = (B,W) |
| 3 | `Z3 = A2 @ W3 + b3`, `P = softmax(Z3)` | (B,W)·(W,10) = (B,10) |

The rule that makes the shapes work: **the inner dimensions of each product must match**,
so the first dimension of every weight matrix is the width of the layer before it. The bias has
one value per output unit and is broadcast (added to every row) across the batch.

## Parameter shapes and counts

| Param | Shape | W = 16 | W = 64 |
|---|---|---:|---:|
| W1 | (64, W) | 1024 | 4096 |
| b1 | (W,) | 16 | 64 |
| W2 | (W, W) | 256 | 4096 |
| b2 | (W,) | 16 | 64 |
| W3 | (W, 10) | 160 | 640 |
| b3 | (10,) | 10 | 10 |
| **Total** | | **1482** | **8970** |

General formula: `64W + W + W² + W + 10W + 10 = W² + 76W + 10`.
Because the `W2` term is quadratic, going from W=16 to W=64 (4x wider) costs about 6x the parameters.
`W2` grows 16x (256 → 4096).

## Initialisation

- Weights are drawn from `N(0, 0.01)`, with **0.01 as the standard deviation** (`rng.normal(0, 0.01)`).
  Drawing random values breaks symmetry. If every weight were equal, every hidden unit would compute the same
  function and get the same gradient, so the units would never become different from each other.
- Biases are set to **0**. Symmetry is already broken by the weights, so the biases don't need randomness.
- Consequence: the weights are tiny, so the logits `Z3` come out almost exactly 0 and the softmax is close to
  uniform, giving each class a probability of about 0.1. That's why the untrained loss in Task 4 should be about `-ln(0.1) = ln(10) ≈ 2.3026`.
- Trade-off: such a small std shrinks the signal at every layer, so gradients for the early layers start small and
  training with a small learning rate (0.001) will be slow. Task 6 will show this.
