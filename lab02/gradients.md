# Lab 2 — Task 5: Deriving the gradients by hand

Write this on paper in order. Each step only uses the one before it (chain rule, from the loss backwards).

## 0. What we start from (the forward pass from Task 3)

```
Z1 = X·W1 + b1      A1 = ReLU(Z1)
Z2 = A1·W2 + b2     A2 = ReLU(Z2)
Z3 = A2·W3 + b3     P  = softmax(Z3)
L  = -(1/B) Σ_i Σ_k Y_ik · log P_ik
```

Notation: `dM` means ∂L/∂M. **`dM` always has the same shape as `M`.** That rule tells you where the
transposes go.

## 1. Softmax + cross-entropy together: dZ3

Work with one sample first (drop the 1/B and the index i):

```
L = -Σ_k y_k log p_k          p_k = e^{z_k} / Σ_j e^{z_j}

log p_k = z_k - log Σ_j e^{z_j}

∂ log p_k / ∂ z_m = δ_km - p_m            (δ_km = 1 if k = m, else 0)

∂L/∂z_m = -Σ_k y_k (δ_km - p_m)
        = -y_m + p_m Σ_k y_k
        = p_m - y_m                       (one-hot: Σ_k y_k = 1)
```

Averaging over the batch brings back the 1/B:

```
dZ3 = (P - Y) / B                                    shape (B,10)
```

Meaning: the error signal is simply "predicted probability minus truth". For the true class it is
`p - 1 < 0` (push the score up); for every other class it is `p > 0` (push the score down).

## 2. Output layer: dW3, db3, dA2

Write the layer element by element: `z3_ik = Σ_j a2_ij · w3_jk + b3_k`

```
∂z3_ik / ∂w3_jk = a2_ij   ->  dW3_jk = Σ_i a2_ij · dZ3_ik   ->  dW3 = A2ᵀ · dZ3     (W,B)·(B,10) = (W,10)
∂z3_ik / ∂b3_k  = 1       ->  db3_k  = Σ_i dZ3_ik           ->  db3 = sum of dZ3 over rows   (10,)
∂z3_ik / ∂a2_ij = w3_jk   ->  dA2_ij = Σ_k dZ3_ik · w3_jk   ->  dA2 = dZ3 · W3ᵀ     (B,10)·(10,W) = (B,W)
```

Shape check: `dW3` must be `(W,10)` like `W3`. The only way to get `(W,10)` from `A2 (B,W)` and
`dZ3 (B,10)` is `A2ᵀ·dZ3`. The sum over the batch index `i` happens inside that matrix product.

## 3. ReLU: dZ2

```
A2 = max(0, Z2)      ReLU'(z) = 1 if z > 0, else 0

dZ2 = dA2 ⊙ 1[Z2 > 0]                                shape (B,W)
```

`⊙` is element-wise multiplication. ReLU is a gate: the gradient passes where the unit was active and is
blocked where it was off. (The derivative at exactly z = 0 is undefined; we take 0.)

## 4. Hidden layer 2: same pattern as step 2

```
dW2 = A1ᵀ · dZ2        (W,B)·(B,W) = (W,W)
db2 = Σ_rows dZ2       (W,)
dA1 = dZ2 · W2ᵀ        (B,W)·(W,W) = (B,W)
```

## 5. ReLU again: dZ1

```
dZ1 = dA1 ⊙ 1[Z1 > 0]                                (B,W)
```

## 6. Hidden layer 1

```
dW1 = Xᵀ · dZ1         (64,B)·(B,W) = (64,W)
db1 = Σ_rows dZ1       (W,)
```

(No `dX` is needed: X is data, not a parameter.)

## Summary box (the whole backward pass)

```
dZ3 = (P - Y)/B
dW3 = A2ᵀ dZ3     db3 = Σ dZ3     dA2 = dZ3 W3ᵀ
dZ2 = dA2 ⊙ [Z2>0]
dW2 = A1ᵀ dZ2     db2 = Σ dZ2     dA1 = dZ2 W2ᵀ
dZ1 = dA1 ⊙ [Z1>0]
dW1 = Xᵀ dZ1      db1 = Σ dZ1
```

Every layer repeats one pattern: **weight gradient = (layer input)ᵀ · (error at output); error for the
layer below = (error) · Wᵀ**. Note there are **two matrix products per layer** in backward, and one in forward.
Task 7 will measure this.

## 7. Numerical gradient check

For one weight w, nudge it both ways and measure the slope of the loss (central difference):

```
numeric = [ L(w + h) - L(w - h) ] / (2h)        h = 1e-5
rel_err = |analytic - numeric| / (|analytic| + |numeric|)      must be < 1e-6
```

The central difference has error O(h²) instead of O(h) for a one-sided difference, so it is far more accurate.

Why the check runs on **one sample**: float64 stores a loss of about 2.3 to about 4e-16, so the numerical slope always
carries rounding noise of about 4e-16 / 2e-5 ≈ 1e-11 (measured: about 1e-11). A batch mean divides every gradient by B,
so with B = 32 many gradients sit near 1e-6, and the noise alone makes their relative error larger than 1e-6.
With one sample, the median gradient is about 7× larger (9.7e-5 vs 1.4e-5 measured) and the check measures the formula, not the rounding.
A second limit: if `w ± h` makes a ReLU input cross 0 (a "kink"), the loss isn't smooth there and the
two slopes legitimately differ.
