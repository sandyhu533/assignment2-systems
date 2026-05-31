# Backward / Saved-Tensor Cheat Sheet

For each op: forward formula, gradient formula, what PyTorch saves for backward,
and memory implication. Aimed at LLM serving/training: understanding **which
tensors live until backward** is the key to reasoning about activation memory
(checkpointing, FlashAttention, ZeRO-Offload, etc.).

Notation:
- `gy` = grad w.r.t. output (incoming from downstream)
- `gx` = grad w.r.t. input (outgoing to upstream)
- "saves: X" = X is held in `_saved_*` on the backward Node until backward runs
- "🟢" = free / negligible, "🟡" = small (statistics, scalars), "🔴" = big

---

## Rule of thumb

| Backward formula contains... | Op must save... |
|---|---|
| only `gy` (e.g. `gx = gy`) | nothing |
| `x` (input) | input |
| `y` (output) | output |
| both | both |
| metadata only (shape, dim, dtype) | nothing (cheap) |

**Inputs vs outputs**: if both work, PyTorch usually saves the **output**
(enables in-place ops to reuse input memory). If only inputs work, must save input.

---

## 1. Matmul family — saves all inputs that the OTHER one needs grad for

| Op | Forward | Backward | Saves | Memory |
|---|---|---|---|---|
| `mm(A, B)` | `Y = A @ B` | `gA = gY @ B^T`, `gB = A^T @ gY` | A, B | 🔴🔴 |
| `bmm(A, B)` | batched matmul | same per-batch | A, B | 🔴🔴 |
| `matmul(A, B)` | general matmul | same | A, B | 🔴🔴 |
| `addmm(c, A, B)` | `Y = c + A @ B` | bias grad = `sum(gY)`, no save for c | A, B | 🔴🔴 |
| `linear(x, W, b)` | `Y = x @ W^T + b` | same as addmm | x, W | 🔴🔴 |
| `einsum(...)` | depends | compiles to bmm/mm | inputs | 🔴🔴 |

**Optimization**: if `A` doesn't require grad → don't save `B`. If `B` doesn't
require grad → don't save `A`. (LoRA / frozen layers benefit hugely.)

**Output is never saved** — gradient formulas don't reference it.

---

## 2. Element-wise binary — only saves when needed for the chain rule

| Op | Forward | Backward | Saves | Memory |
|---|---|---|---|---|
| `add` | `y = a + b` | `ga = gy`, `gb = gy` | **nothing** | 🟢 |
| `sub` | `y = a - b` | `ga = gy`, `gb = -gy` | **nothing** | 🟢 |
| `neg` | `y = -x` | `gx = -gy` | nothing | 🟢 |
| `mul` | `y = a * b` | `ga = gy * b`, `gb = gy * a` | a, b | 🔴🔴 |
| `div` | `y = a / b` | `ga = gy / b`, `gb = -gy * a / b²` | a, b | 🔴🔴 |
| `pow` (const k) | `y = x^k` | `gx = k * x^(k-1) * gy` | x, k | 🔴 |

**Killer insight**: `add` is **free**! That's why residual connections cost
nothing in activation memory. `mul` and `div` cost double — both inputs saved.

---

## 3. Unary functions — output-saving vs input-saving

| Op | Forward | Backward (closed form) | Saves | Memory |
|---|---|---|---|---|
| `exp` | `y = exp(x)` | `gx = y * gy` | **output** | 🔴 |
| `log` | `y = log(x)` | `gx = gy / x` | input | 🔴 |
| `sqrt` | `y = sqrt(x)` | `gx = gy / (2y)` | **output** | 🔴 |
| `reciprocal` | `y = 1/x` | `gx = -y² * gy` | output | 🔴 |
| `sin` | `y = sin(x)` | `gx = cos(x) * gy` | input | 🔴 |
| `cos` | `y = cos(x)` | `gx = -sin(x) * gy` | input | 🔴 |

**Memorize this**: `exp` saves output. Used by softmax's exp step → softmax
already has the exp values cached.

---

## 4. Activations — most save output (good for in-place)

| Op | Backward identity | Saves | Memory note |
|---|---|---|---|
| `relu` | `gx = gy * (y > 0)` | **output** | 🔴 (1 copy of activations) |
| `sigmoid` | `gx = gy * y * (1-y)` | **output** | 🔴 |
| `tanh` | `gx = gy * (1 - y²)` | **output** | 🔴 |
| `softmax` | `gx = y * (gy - sum(y*gy))` | **output** | 🔴 (the L×L matrix in attention!) |
| `log_softmax` | similar | output | 🔴 |
| `silu` (swish) | `y = x * σ(x)`, grad needs both | **input** | 🔴 |
| `gelu` | involves erf/tanh, grad needs x | **input** | 🔴 |
| `leaky_relu` | gated by `x > 0` | input | 🔴 |
| `dropout` | `gx = gy * mask / (1-p)` | mask (bool!) | 🟡 (1 byte/elem, not 4) |

**Why dropout is cheap**: mask is bool, 8× smaller than fp32. But total activations
still grow by 1/8 per dropout layer.

**Why softmax in attention is expensive**: saves the L×L output. This is the
biggest single activation in attention, and the one FlashAttention eliminates.

---

## 5. Reductions — `sum` is free, `max` is not

| Op | Backward | Saves | Memory |
|---|---|---|---|
| `sum(x)` | `gx = gy.expand_as(x)` | **nothing** (just shape) | 🟢 |
| `mean(x)` | `gx = gy.expand_as(x) / N` | shape + N | 🟢 |
| `amax(x, dim)` | grad → max positions (split on ties) | input + output | 🔴 + 🟡 |
| `amin(x, dim)` | grad → min positions | input + output | 🔴 + 🟡 |
| `argmax/argmin` | non-differentiable | n/a | n/a |
| `prod(x)` | `gx_i = y / x_i * gy` (NaN if any x=0) | input + output | 🔴 + 🟡 |
| `norm(x, p=2)` | `gx = x / y * gy` | input + output | 🔴 + 🟡 |
| `var/std` | depends on unbiased flag | input + stats | 🔴 + 🟡 |

**Killer insight**: `sum` and `mean` are **free** in activation memory. So
`loss = something.sum()` adds zero memory. Pooling operations that decompose
to sum/mean are also free.

---

## 6. Norm layers — save input + small stats

| Op | What's saved | Memory |
|---|---|---|
| `LayerNorm` | input, weight, mean, **rstd** (1/σ) | 🔴 input + 🟡🟡 stats |
| `RMSNorm` | input, weight, rrms | 🔴 input + 🟡 |
| `BatchNorm` | input, weight, saved_mean, saved_invstd, running stats | 🔴 input + 🟡 |
| `GroupNorm` | input, weight, mean, rstd | 🔴 input + 🟡 |

**LLM-relevant**: every transformer block has 2 LayerNorm/RMSNorm → 2× input
copies saved per block. For a 32-layer model, that's 64 copies of `(B, L, D)`.
This is why some implementations fuse the residual + norm together.

---

## 7. Shape ops — entirely free

| Op | Backward | Saves |
|---|---|---|
| `view`, `reshape` | reverse shape | shape only |
| `transpose`, `permute`, `.T` | reverse perm | perm only |
| `squeeze`, `unsqueeze` | reverse | dim only |
| `expand`, `broadcast_to` | sum over broadcast dims | original shape |
| `contiguous` | identity | nothing |
| `as_strided` | reverse stride | stride info |

**All free in activation memory.** They just record metadata so backward can
undo the shape change. (`expand` is interesting: forward is a view, backward
is `sum` over the expanded dim.)

---

## 8. Indexing / gather / scatter

| Op | Backward | Saves | Memory |
|---|---|---|---|
| `x[i:j]` slice | pad zeros | slice info | 🟢 |
| `x[mask]` boolean index | scatter back via mask | mask | 🟡 (bool) |
| `gather(x, dim, idx)` | `scatter_add` | idx, x.shape | 🟡 idx only |
| `index_select(x, dim, idx)` | scatter_add | idx, x.shape | 🟡 |
| `embedding(W, ids)` | `scatter_add` into grad_W | ids | 🟡 (int64 ids) |
| `scatter(x, dim, idx, src)` | inverse gather | idx | 🟡 |

**Killer insight**: `embedding` lookup is **almost free** in activation memory
(just save the ids). The big memory cost of embedding is the parameter W, not
its activations. This is why vocab parallelism (split W across GPUs) is more
important than splitting activations.

---

## 9. Concat / split / stack

| Op | Backward | Saves |
|---|---|---|
| `cat([a,b,...], dim)` | split | input shapes (no tensors!) |
| `split(x, sizes, dim)` | cat | split info |
| `chunk` | cat | n, dim |
| `stack` | unstack | shape info |
| `unbind` | stack | dim |

**All free.** Tensor manipulation in PyTorch is gloriously cheap for autograd.

---

## 10. Masking / where

| Op | Backward | Saves | Memory |
|---|---|---|---|
| `where(cond, a, b)` | `ga = gy * cond`, `gb = gy * !cond` | **cond only** | 🟡 (bool) |
| `masked_fill(x, mask, val)` | grad zeroed at mask | mask | 🟡 |
| `masked_select(x, mask)` | scatter back | mask | 🟡 |

**Why `where` is cheap**: doesn't need `a` or `b` values — gradient just gated
by the boolean. This is what saves attention's causal mask: it's only `L×L` bool,
1 MB for `L=1024`, not 32 MB.

---

## 11. Cast / dtype

| Op | Backward | Saves |
|---|---|---|
| `x.float()` / `x.to(dtype)` | cast back | dtype only |
| `x.type_as(y)` | cast back | dtype |
| `x.detach()` | breaks graph | n/a |

🟢 Free. Mixed precision (`x.half()` followed by op) costs 0 in saved memory;
the cost is in *allocating* the cast tensor (which is forward-time memory).

---

## 12. Losses

| Op | Saves |
|---|---|
| `mse_loss(a, b)` | a, b |
| `cross_entropy(logits, target)` | logits (or output), target |
| `nll_loss(log_probs, target)` | target |
| `binary_cross_entropy_with_logits` | logits, target |

**LLM-relevant**: `cross_entropy(logits, target)` where logits is `(B, L, V)` (V=vocab)
saves the entire logits tensor → can be 10+ GB on big-vocab models. This is
why people use **chunked cross-entropy** (e.g., Liger Kernel's fused CE).

---

## 13. Fused / specialized

| Op | Saves | vs naive |
|---|---|---|
| `F.scaled_dot_product_attention` | Q, K, V, O, logsumexp | 🟢🟢🟢 vs L×L for naive |
| `F.layer_norm` | input, weight, mean, rstd | same as manual |
| `F.softmax` | output only | vs manual (amax + sub + exp + sum + div) = 3 L×L copies! |
| `F.gelu` (fused) | input | vs manual gelu (many ops) |

**Always prefer fused ops in PyTorch** — they save activations by structure, not just kernel launches.

---

## 14. Misc useful

| Op | Saves | Notes |
|---|---|---|
| `clone()` | nothing | grad passes through |
| `detach()` | n/a | breaks graph entirely |
| `t.fill_(v)`, `t.zero_()` (in-place) | input *can't* be saved (modified) | autograd errors if saved |
| `torch.no_grad()` block | none | forward doesn't build graph |
| `requires_grad_(False)` on leaf | depends on op | partner-input optimization kicks in |

---

## How to use this sheet during code review / interview

**"How much activation memory does this layer cost?"**

1. Walk through each op forward.
2. For each op, look up its saved tensors in the table.
3. Dedupe by storage (views share memory).
4. Sum the unique storage sizes.

**Example: transformer FFN `y = W2 @ gelu(W1 @ x)`**

| Op | Saves | Size |
|---|---|---|
| `W1 @ x` | x (B,L,D), W1 (D,4D) | x: `B·L·D`, W1: param (already counted) |
| `gelu(...)` | input (B,L,4D) | `B·L·4D` |
| `W2 @ ...` | gelu_out (B,L,4D), W2 (4D,D) | `B·L·4D`, W2: param |

→ Activations: `B·L·D + 2·B·L·4D = 9·B·L·D` per FFN block. The gelu input is
saved **twice** (once for gelu backward, once for W2 backward) but if storage
is the same it dedupes to once. Reality: gelu allocates a new tensor for output,
so the input and output are *different* storage → both saved → really `9·B·L·D`.

**This is the activation memory FFN saves and what tools like checkpoint cut.**

---

## What this sheet won't tell you

- **Parameter memory** — `W` is just held by the module, not via `_saved_*`.
- **Optimizer state** — Adam's m, v are separate.
- **Allocator fragmentation** — caching allocator can hold blocks beyond what's "live".
- **In-flight comm buffers** — TP/PP/FSDP allocate extra for collectives.

For those, you need `torch.cuda.memory_summary()` or a profiler. This sheet
covers only what autograd holds for backward.
