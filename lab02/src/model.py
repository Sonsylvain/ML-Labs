"""The from-scratch network: 64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax.

Convention: a batch X has shape (batch, features), so each layer is z = X @ W + b
with W of shape (in, out) and b of shape (out,), broadcast over the batch.
"""

import numpy as np

N_INPUT = 64
N_OUTPUT = 10
INIT_STD = 0.01  # N(0, 0.01): 0.01 is the standard deviation


def layer_sizes(width: int) -> list[int]:
    return [N_INPUT, width, width, N_OUTPUT]


def init_params(width: int, rng: np.random.Generator) -> dict:
    """Weights ~ N(0, 0.01), biases = 0. Keys W1,b1,W2,b2,W3,b3."""
    sizes = layer_sizes(width)
    params = {}
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:]), start=1):
        params[f"W{i}"] = rng.normal(0.0, INIT_STD, size=(n_in, n_out))
        params[f"b{i}"] = np.zeros(n_out)
    return params


def count_params(params: dict) -> int:
    return sum(p.size for p in params.values())


def relu(Z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, Z)


def softmax(Z: np.ndarray) -> np.ndarray:
    """Row-wise softmax. Subtracting the row max leaves the result unchanged
    (it cancels in the ratio) but keeps exp() from overflowing."""
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def forward(params: dict, X: np.ndarray) -> tuple[np.ndarray, dict]:
    """X (B,64) -> P (B,10). Also returns the cache of intermediates needed by backprop."""
    Z1 = X @ params["W1"] + params["b1"]   # (B,W)
    A1 = relu(Z1)                          # (B,W)
    Z2 = A1 @ params["W2"] + params["b2"]  # (B,W)
    A2 = relu(Z2)                          # (B,W)
    Z3 = A2 @ params["W3"] + params["b3"]  # (B,10)
    P = softmax(Z3)                        # (B,10)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "P": P}
    return P, cache


EPS = 1e-12  # keeps log() finite if a probability underflows to exactly 0


def cross_entropy(P: np.ndarray, Y: np.ndarray) -> float:
    """Mean over the batch of -sum_k Y_k log P_k. With one-hot Y this is -log(P of the true class)."""
    return float(-np.mean(np.sum(Y * np.log(P + EPS), axis=1)))


def backward(params: dict, cache: dict, Y: np.ndarray) -> dict:
    """Gradients of the mean cross-entropy w.r.t. every parameter (chain rule, see gradients.md)."""
    B = Y.shape[0]
    dZ3 = (cache["P"] - Y) / B                 # (B,10)  softmax + cross-entropy combined
    dW3 = cache["A2"].T @ dZ3                  # (W,B)·(B,10) -> (W,10)
    db3 = dZ3.sum(axis=0)                      # (10,)
    dA2 = dZ3 @ params["W3"].T                 # (B,10)·(10,W) -> (B,W)
    dZ2 = dA2 * (cache["Z2"] > 0)              # ReLU passes gradient only where Z2 > 0
    dW2 = cache["A1"].T @ dZ2                  # (W,W)
    db2 = dZ2.sum(axis=0)                      # (W,)
    dA1 = dZ2 @ params["W2"].T                 # (B,W)
    dZ1 = dA1 * (cache["Z1"] > 0)              # (B,W)
    dW1 = cache["X"].T @ dZ1                   # (64,B)·(B,W) -> (64,W)
    db1 = dZ1.sum(axis=0)                      # (W,)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
