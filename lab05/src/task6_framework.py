"""Task 6: inspect a small PyTorch model's graphs, time eager vs compiled/traced, and measure contiguity cost."""

import statistics
import time

import pandas as pd
import torch
import torch.fx

from common import RESULTS_DIR, SEED

torch.set_num_threads(1)  # one thread: timings are stable and comparable
BATCHES = [1, 64, 4096]
WARMUP = 50
BLOCKS, CALLS = 15, 200


def make_model():
    torch.manual_seed(SEED)
    return torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1), torch.nn.Sigmoid())


def autograd_tree(fn, depth=0, seen=None) -> list[str]:
    """Walk grad_fn -> next_functions: the backward graph autograd recorded during forward."""
    seen = set() if seen is None else seen
    if fn is None:
        return []
    name = type(fn).__name__
    if fn in seen:
        return ["  " * depth + name + " (shared)"]
    seen.add(fn)
    lines = ["  " * depth + name]
    for nxt, _ in fn.next_functions:
        lines += autograd_tree(nxt, depth + 1, seen)
    return lines


def per_call_us(fn, x) -> float:
    """Median over BLOCKS of the mean time of CALLS calls, after WARMUP calls."""
    with torch.no_grad():
        for _ in range(WARMUP):
            fn(x)
        blocks = []
        for _ in range(BLOCKS):
            t0 = time.perf_counter()
            for _ in range(CALLS):
                fn(x)
            blocks.append((time.perf_counter() - t0) / CALLS)
    return statistics.median(blocks) * 1e6


def op_ms(fn, repeats=30) -> float:
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def describe(name, t) -> str:
    return (f"  {name:<22} shape {str(tuple(t.shape)):<13} dtype {str(t.dtype):<14} "
            f"stride {str(t.stride()):<10} contiguous {t.is_contiguous()}")


if __name__ == "__main__":
    model = make_model().eval()
    example = torch.randn(64, 2)
    report = []

    # ---- 1. graphs ----
    out = model(example).mean()
    report += ["== autograd backward graph (recorded during forward of model(x).mean()) =="]
    report += autograd_tree(out.grad_fn)
    report += ["", "== torch.fx symbolic trace (forward graph) ==", str(torch.fx.symbolic_trace(model).graph)]
    traced = torch.jit.trace(model, example)
    scripted = torch.jit.script(model)
    report += ["== torch.jit.trace graph ==", str(traced.graph)]

    # ---- 2. torch.compile, if this machine supports it ----
    compiled = torch.compile(lambda x: model(x))
    try:
        with torch.no_grad():
            compiled(example)
        compile_status = "torch.compile (inductor): OK"
    except Exception as e:  # inductor needs a C++ compiler at run time
        compiled = None
        compile_status = f"torch.compile (inductor): UNSUPPORTED here -> {type(e).__name__}: " + \
                         str(e).strip().splitlines()[-1 if "Cxx" not in str(e) else 1].strip()
    report += ["== torch.compile ==", compile_status, ""]
    print(compile_status)

    # ---- 3. eager vs traced / scripted (/ compiled) ----
    variants = {"eager": model, "jit.trace": traced, "jit.script": scripted}
    if compiled is not None:
        variants["torch.compile"] = compiled
    rows = []
    for B in BATCHES:
        x = torch.randn(B, 2)
        with torch.no_grad():
            ref = model(x)
            for name, fn in variants.items():
                assert torch.allclose(fn(x), ref, atol=1e-6), name
        for name, fn in variants.items():
            rows.append({"batch": B, "variant": name, "us_per_call": per_call_us(fn, x)})
    timing = pd.DataFrame(rows)
    eager = timing[timing.variant == "eager"].set_index("batch").us_per_call
    timing["speedup_vs_eager"] = timing.apply(lambda r: eager[r.batch] / r.us_per_call, axis=1)
    timing.to_csv(RESULTS_DIR / "execution_timings.csv", index=False, float_format="%.4f")
    print(timing.to_string(index=False, float_format="%.3f"))

    # ---- 4. tensor shape, dtype, contiguity ----
    torch.manual_seed(SEED)
    a = torch.randn(4096, 4096)
    at = a.t()            # a view: same memory, strides swapped
    ac = at.contiguous()  # a real copy laid out row by row
    report += ["== tensor layout ==", describe("a", a), describe("a.t() (view)", at),
               describe("a[:, ::2] (view)", a[:, ::2]), describe("a.t().contiguous()", ac),
               f"  a.t() shares storage with a: {at.data_ptr() == a.data_ptr()}"]
    try:
        at.view(-1)
    except RuntimeError as e:
        report.append(f"  a.t().view(-1) -> RuntimeError: {str(e).splitlines()[0][:90]}...")
    report.append("")

    ops = {"row sum  x.sum(dim=1)": lambda x: x.sum(dim=1),
           "elementwise  x * 2 + 1": lambda x: x * 2 + 1,
           "copy  x.clone()": lambda x: x.clone(),
           "matmul  x @ v (4096x4096 . 4096)": lambda x: x @ v}
    v = torch.randn(4096)
    rows = []
    for name, op in ops.items():
        t_c, t_nc = op_ms(lambda: op(ac)), op_ms(lambda: op(at))
        rows.append({"operation": name, "contiguous_ms": t_c, "non_contiguous_ms": t_nc,
                     "slowdown": t_nc / t_c})
    # A strided slice has gaps between the elements it uses, unlike a transpose
    s_nc = a[:, ::2]
    s_c = s_nc.contiguous()
    for name, op in list(ops.items())[:2]:
        t_c, t_nc = op_ms(lambda: op(s_c)), op_ms(lambda: op(s_nc))
        rows.append({"operation": f"{name}  [a[:, ::2]]", "contiguous_ms": t_c,
                     "non_contiguous_ms": t_nc, "slowdown": t_nc / t_c})
    rows.append({"operation": "make contiguous  a.t().contiguous()",
                 "contiguous_ms": float("nan"), "non_contiguous_ms": op_ms(lambda: at.contiguous()),
                 "slowdown": float("nan")})
    contig = pd.DataFrame(rows)
    contig.to_csv(RESULTS_DIR / "contiguity_timings.csv", index=False, float_format="%.4f")
    print(contig.to_string(index=False, float_format="%.3f"))

    text = "\n".join(report)
    (RESULTS_DIR / "graphs_and_layout.txt").write_text(text + "\n")
    print(text)
