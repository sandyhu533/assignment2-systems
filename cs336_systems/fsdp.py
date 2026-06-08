import math
import torch
import torch.distributed as dist
import torch.nn as nn

def _all_gather_full(shard, full_shape, world_size, compute_dtype):
    s = shard
    if compute_dtype is not None:
        s = s.to(compute_dtype)
    gathered = [torch.empty_like(s) for _ in range(world_size)]
    dist.all_gather(gathered, s.contiguous())
    return torch.cat(gathered, dim=0)[: full_shape[0]].reshape(full_shape)


def _reduce_scatter_grad(grad, full_shape, per, world_size):
    grad = grad.to(torch.float32)
    n0 = full_shape[0]
    pad = per * world_size - n0
    if pad:
        grad = torch.cat([grad, grad.new_zeros(pad, *grad.shape[1:])], dim=0)
    in_list = [c.contiguous() for c in grad.chunk(world_size, dim=0)]
    out = torch.empty_like(in_list[0])
    dist.reduce_scatter(out, in_list, op=dist.ReduceOp.SUM)
    return (out / world_size).to(torch.float32)


class _ShardedLinear(torch.autograd.Function):
    """y = x @ W.T, where W is gathered from `shard` INSIDE this node.

    forward:  gather W -> y -> save only (x, shard). W is freed when forward
              returns (not a saved tensor, not referenced downstream).
    backward: re-gather W, compute grad_x = grad_y @ W, grad_W = grad_y^T @ x,
              reduce-scatter grad_W into the fp32 shard grad. W freed again.
    Peak: only the shard + activations are retained fwd->bwd; the full W is
          transient inside fwd and inside bwd, never spanning both.
    """

    @staticmethod
    def forward(ctx, x, shard, handle):
        ctx.handle = handle
        ctx.save_for_backward(x, shard)
        W = _all_gather_full(shard, handle.full_shape, handle.world_size, handle.compute_dtype)
        cdt = handle.compute_dtype
        xc = x.to(cdt) if cdt is not None else x
        y = xc @ W.T
        return y  # W goes out of scope here

    @staticmethod
    def backward(ctx, grad_y):
        x, shard = ctx.saved_tensors
        handle = ctx.handle
        cdt = handle.compute_dtype
        W = _all_gather_full(shard, handle.full_shape, handle.world_size, cdt)
        xc = x.to(cdt) if cdt is not None else x
        gy = grad_y.to(cdt) if cdt is not None else grad_y
        grad_x = gy @ W                                   # [.., out] @ [out,in]
        grad_W = gy.reshape(-1, gy.shape[-1]).T @ xc.reshape(-1, xc.shape[-1])
        shard_grad = _reduce_scatter_grad(grad_W, handle.full_shape, handle.per, handle.world_size)
        grad_x = grad_x.to(x.dtype)
        return grad_x, shard_grad, None


class _ShardedEmbedding(torch.autograd.Function):
    """y = W[idx], W gathered from `shard` inside this node."""

    @staticmethod
    def forward(ctx, idx, shard, handle):
        ctx.handle = handle
        ctx.save_for_backward(idx, shard)
        W = _all_gather_full(shard, handle.full_shape, handle.world_size, handle.compute_dtype)
        return W[idx]

    @staticmethod
    def backward(ctx, grad_y):
        idx, shard = ctx.saved_tensors
        handle = ctx.handle
        grad_W = torch.zeros(handle.full_shape, dtype=torch.float32, device=shard.device)
        grad_W.index_add_(0, idx.reshape(-1), grad_y.reshape(-1, grad_y.shape[-1]).to(torch.float32))
        shard_grad = _reduce_scatter_grad(grad_W, handle.full_shape, handle.per, handle.world_size)
        return None, shard_grad, None


class _ShardedHandle:
    def __init__(self, fsdp, mod, pname, shard_param, kind):
        self.world_size = fsdp.world_size
        self.compute_dtype = fsdp.compute_dtype
        self.mod = mod
        self.pname = pname
        self.shard = shard_param
        self.full_shape = shard_param._fsdp_full_shape
        self.per = shard_param._fsdp_per
        self.kind = kind  # "linear" | "embedding"


class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        from cs336_basics.model import Embedding, Linear
        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self._handles = []
        self._replicated = []
        self._saved_repl = []

        for mod in self.module.modules():
            if isinstance(mod, Linear):
                self._wrap(mod, "weight", "linear", _ShardedLinear)
            elif isinstance(mod, Embedding):
                self._wrap(mod, "weight", "embedding", _ShardedEmbedding)
            else:
                for pname, p in list(mod.named_parameters(recurse=False)):
                    self._replicated.append(p)

    def _wrap(self, mod, pname, kind, fn):
        p = getattr(mod, pname)
        self._shard_param(mod, pname, p)
        h = _ShardedHandle(self, mod, pname, getattr(mod, pname), kind)
        self._handles.append(h)
        # Replace the module's forward to route through the sharded autograd op.
        if kind == "linear":
            def forward(x, _h=h, _fn=fn):
                return _fn.apply(x, _h.shard, _h)
        else:
            def forward(idx, _h=h, _fn=fn):
                return _fn.apply(idx, _h.shard, _h)
        mod.forward = forward

    def _shard_param(self, mod, pname, p):
        full = p.data
        full_shape = full.shape
        n0 = full_shape[0]
        per = math.ceil(n0 / self.world_size)
        pad = per * self.world_size - n0
        if pad:
            full = torch.cat([full, full.new_zeros(pad, *full_shape[1:])], dim=0)
        shard = full[self.rank * per:(self.rank + 1) * per].clone()
        new_p = nn.Parameter(shard, requires_grad=p.requires_grad)
        new_p._fsdp_full_shape = full_shape
        new_p._fsdp_per = per
        setattr(mod, pname, new_p)

    def _cast_replicated(self, dt):
        self._saved_repl = []
        for p in self._replicated:
            self._saved_repl.append((p, p.data))
            p.data = p.data.to(dt)

    def _restore_replicated(self):
        for p, data in self._saved_repl:
            p.data = data
        self._saved_repl = []

    def forward(self, *args, **kwargs):
        if self.compute_dtype is not None:
            self._cast_replicated(self.compute_dtype)
        out = self.module(*args, **kwargs)
        if self.compute_dtype is not None:
            self._restore_replicated()
        return out

    def finish_gradient_synchronization(self):
        # Sharded grads were reduce-scattered inside the autograd backward and
        # written by autograd onto each shard leaf's .grad. Fill zeros if unused.
        for h in self._handles:
            if h.shard.requires_grad and h.shard.grad is None:
                h.shard.grad = torch.zeros_like(h.shard.data, dtype=torch.float32)

        for p in self._replicated:
            if p.grad is None:
                continue
            p.grad = p.grad.to(torch.float32)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= self.world_size


def fsdp_gather_full_params(fsdp_model: FSDP) -> dict[str, torch.Tensor]:
    out = {}
    sharded_full = {}
    for h in fsdp_model._handles:
        full = _all_gather_full(h.shard.data.to(torch.float32), h.full_shape, fsdp_model.world_size, None)
        sharded_full[h.shard] = full

    for name, p in fsdp_model.module.named_parameters():
        if p in sharded_full:
            out[name] = sharded_full[p]
        else:
            out[name] = p.data.to(torch.float32).clone()
    return out