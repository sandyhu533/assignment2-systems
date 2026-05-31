"""
Autograd inspection utilities — for understanding what PyTorch keeps alive
between forward and backward, and in what order backward actually runs.

Usage sketch:
    from autograd_inspect import print_graph, print_saved_activations, trace_backward

    out = model(x)
    print_graph(out)
    print_saved_activations(out)
    trace_backward(out)
    out.sum().backward()
"""
import torch
from collections import defaultdict


def _node_name(node):
    """Short identifier for a grad_fn node: type + last 4 hex of id."""
    return f"{type(node).__name__}@{id(node) & 0xFFFF:04x}"


def _saved_tensors_on(node):
    """Return list of (attr_name, tensor) for tensors saved on this node.

    Built-in C++ Nodes expose saved tensors via `_saved_*` named attrs.
    Custom torch.autograd.Function uses `saved_tensors` instead.
    """
    out = []
    for attr in dir(node):
        if not attr.startswith("_saved"):
            continue
        try:
            val = getattr(node, attr)
        except (RuntimeError, AttributeError):
            continue
        if isinstance(val, torch.Tensor):
            out.append((attr, val))
    # custom Functions
    if hasattr(node, "saved_tensors"):
        try:
            for i, t in enumerate(node.saved_tensors):
                out.append((f"saved_tensors[{i}]", t))
        except RuntimeError:
            pass
    return out


def _walk(root_grad_fn):
    """BFS over the autograd graph starting at a tensor's grad_fn.
    Returns (nodes_in_topo_order, edges) where edges[child] = [parent, ...].
    """
    seen = {}  # id(node) -> node
    order = []
    edges = defaultdict(list)
    stack = [root_grad_fn]
    while stack:
        n = stack.pop()
        if n is None or id(n) in seen:
            continue
        seen[id(n)] = n
        order.append(n)
        for parent, _ in n.next_functions:
            if parent is not None:
                edges[id(n)].append(parent)
                stack.append(parent)
    return order, edges


# ---------------------------------------------------------------------------
# Tool 1: print the autograd graph topology
# ---------------------------------------------------------------------------
def print_graph(tensor, max_depth=10):
    """Recursively print the autograd DAG rooted at `tensor.grad_fn`.

    Shared nodes (DAG, not tree) are printed once and then referenced by name.
    """
    if tensor.grad_fn is None:
        print(f"<leaf or no grad> {tensor.shape}")
        return

    visited = {}  # id(node) -> name

    def _print(node, depth, edge_label=""):
        if node is None:
            print("  " * depth + f"{edge_label}<None>")
            return
        name = _node_name(node)
        if id(node) in visited:
            print("  " * depth + f"{edge_label}{name}  (↑ already shown)")
            return
        visited[id(node)] = name
        saved = _saved_tensors_on(node)
        saved_str = ""
        if saved:
            shapes = ", ".join(f"{n}:{tuple(t.shape)}" for n, t in saved)
            saved_str = f"   saved=[{shapes}]"
        print("  " * depth + f"{edge_label}{name}{saved_str}")
        if depth >= max_depth:
            print("  " * (depth + 1) + "... (max_depth reached)")
            return
        for i, (parent, _) in enumerate(node.next_functions):
            _print(parent, depth + 1, edge_label=f"input[{i}] → ")

    print(f"=== Autograd graph for tensor {tuple(tensor.shape)} ===")
    _print(tensor.grad_fn, 0)


# ---------------------------------------------------------------------------
# Tool 2: list saved activations and their memory footprint
# ---------------------------------------------------------------------------
def print_saved_activations(tensor):
    """Enumerate every tensor saved on the autograd graph and total bytes.

    This is exactly the 'activation memory' that gradient checkpointing
    trades off: each entry here lives until its node's backward runs.
    """
    if tensor.grad_fn is None:
        print("<no graph>")
        return

    nodes, _ = _walk(tensor.grad_fn)
    total_bytes = 0
    # dedupe by data_ptr so we don't double-count when the same storage is
    # saved on multiple nodes (common: views, in-place reuse).
    seen_storage = set()

    print(f"=== Saved activations on graph rooted at {tuple(tensor.shape)} ===")
    print(f"{'node':<22} {'attr':<24} {'shape':<22} {'dtype':<12} {'MB':>10}")
    print("-" * 92)
    for node in nodes:
        for attr, t in _saved_tensors_on(node):
            mb = t.element_size() * t.nelement() / 1024 / 1024
            key = (t.data_ptr(), t.nelement(), t.dtype)
            unique = "" if key in seen_storage else " *"
            if key not in seen_storage:
                total_bytes += t.element_size() * t.nelement()
                seen_storage.add(key)
            print(f"{_node_name(node):<22} {attr:<24} {str(tuple(t.shape)):<22} "
                  f"{str(t.dtype):<12} {mb:>9.4f}{unique}")
    print("-" * 92)
    print(f"Unique storage total: {total_bytes / 1024 / 1024:.4f} MB "
          f"(rows marked * counted once)")


# ---------------------------------------------------------------------------
# Tool 3: trace which backward node fires, in what order, with what gradients
# ---------------------------------------------------------------------------
def trace_backward(tensor, show_values=False):
    """Attach hooks to every node so we print order + grad shapes during backward.

    Call this BEFORE `.backward()`. The hooks fire during `.backward()`.
    """
    if tensor.grad_fn is None:
        print("<no graph>")
        return

    nodes, _ = _walk(tensor.grad_fn)
    counter = {"i": 0}

    def make_hook(node):
        def hook(grad_inputs, grad_outputs):
            counter["i"] += 1
            i = counter["i"]
            name = _node_name(node)
            def fmt(g):
                if g is None:
                    return "None"
                s = f"{tuple(g.shape)}"
                if show_values and g.numel() <= 16:
                    s += f"={g.detach().cpu().tolist()}"
                return s
            gin = ", ".join(fmt(g) for g in grad_inputs)
            gout = ", ".join(fmt(g) for g in grad_outputs)
            print(f"[backward #{i:02d}] {name}")
            print(f"           grad_outputs (from downstream): [{gout}]")
            print(f"           grad_inputs  (to upstream):     [{gin}]")
        return hook

    print(f"=== Registering backward hooks on {len(nodes)} nodes ===")
    for node in nodes:
        node.register_hook(make_hook(node))
    print("Now call .backward() to see execution trace.\n")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn(2, 2, requires_grad=True)
    b = torch.randn(2, 2, requires_grad=True)

    c = a * b
    e = (a * torch.ones(2, 2)) * 5
    d = (c + e).sum()

    print_graph(d)
    print()
    print_saved_activations(d)
    print()
    trace_backward(d, show_values=True)
    d.backward()
