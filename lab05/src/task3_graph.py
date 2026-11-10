"""Tasks 1-3: show that the forward pass builds a graph and that backward accumulates gradients."""

from common import RESULTS_DIR
from value import Value


def describe(order) -> list[str]:
    names = {}
    lines = []
    for node in order:
        names[id(node)] = f"n{len(names)}"
        parents = ", ".join(names[id(p)] for p in node._parents)
        label = f"{node._op}({parents})" if node._op else "leaf"
        lines.append(f"  {names[id(node)]:<4} {label:<18} data {node.data:>9.4f}   grad {node.grad:>9.4f}")
    return lines


if __name__ == "__main__":
    out = []

    # Graph construction: f = (x*y + x)**2 with x used twice
    x, y = Value(2.0), Value(-3.0)
    f = (x * y + x) ** 2
    order = f.backward()
    out += ["f = (x*y + x)**2 at x = 2, y = -3", "topological order (parents before children):"]
    out += describe(order)
    # by hand: f = (x(y+1))^2, df/dx = 2x(y+1)^2 = 16, df/dy = 2x^2(y+1) = -16
    out.append(f"df/dx = {x.grad} (hand: 2x(y+1)^2 = 16)   df/dy = {y.grad} (hand: 2x^2(y+1) = -16)")
    assert (x.grad, y.grad) == (16.0, -16.0)

    # Accumulation: a node reused through two paths must receive both contributions
    a = Value(3.0)
    g = a * a + a  # dg/da = 2a + 1 = 7
    g.backward()
    out += ["", f"g = a*a + a at a = 3: dg/da = {a.grad} (hand: 2a + 1 = 7)"]
    assert a.grad == 7.0
    out.append("  with '=' instead of '+=' the last rule to run would overwrite the others and give 1 or 3")

    # ReLU on both sides of 0
    for v in (-1.5, 0.0, 2.5):
        z = Value(v)
        z.relu().backward()
        out.append(f"relu'({v:+.1f}) = {z.grad}")

    text = "\n".join(out)
    print(text)
    (RESULTS_DIR / "graph_demo.txt").write_text(text + "\n")
