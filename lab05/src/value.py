"""Tasks 1-3: a scalar automatic-differentiation engine.

Every arithmetic operation on Value objects creates a new Value that remembers
its parents and a closure (the local derivative rule) that pushes its gradient
back to them. Calling backward() on the output walks that graph in reverse
topological order.
"""

import math


class Value:
    """One scalar node in the computation graph.

    data      -- the number computed in the forward pass
    grad      -- dL/d(this node), filled in by backward()
    _parents  -- the nodes this one was computed from
    _backward -- local derivative rule: adds this node's grad into its parents' grads
    _op       -- name of the operation, for printing the graph
    """

    __slots__ = ("data", "grad", "_parents", "_backward", "_op")

    def __init__(self, data, _parents=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = _parents
        self._backward = lambda: None  # leaves have no rule
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    # ---- Task 2: operations that build the graph ----

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():  # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():  # d(ab)/da = b, d(ab)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():  # d(x^k)/dx = k x^(k-1)
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():  # 1 where the input was positive, 0 otherwise (0 at x = 0)
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    # Needed for Task 5 (sigmoid output and BCE loss)
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():  # d(e^x)/dx = e^x
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():  # d(ln x)/dx = 1/x
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():  # d(sigma)/dx = sigma (1 - sigma)
            self.grad += s * (1.0 - s) * out.grad
        out._backward = _backward
        return out

    # Composite operations, built from the ones above
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return Value(other) - self

    # ---- Task 3: backward pass ----

    def backward(self):
        # 1. Topological order: every node appears after all of its parents.
        #    Iterative depth-first search, so a deep graph cannot hit Python's recursion limit.
        order, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                stack.extend((p, False) for p in node._parents if id(p) not in seen)

        # 2. Seed: dL/dL = 1.
        self.grad = 1.0
        # 3. Reverse order: a node's grad is complete before it is pushed to its parents.
        #    4. Each rule uses +=, so a node used several times accumulates every path.
        for node in reversed(order):
            node._backward()
        return order


def vsum(values):
    """Sum of a list of Values (the .sum() of the Task 4 expression)."""
    total = Value(0.0)
    for v in values:
        total = total + v
    return total
