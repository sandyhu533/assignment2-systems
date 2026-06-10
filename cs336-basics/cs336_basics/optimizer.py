from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch


def get_cosine_lr(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """Cosine with warmup learning rate scheduler."""
    # First, we linearly warmup for warmup_iters steps.
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    # Then, if it > cosine_cycle_iters, we return min learning rate.
    if it > cosine_cycle_iters:
        return min_learning_rate
    # Else, we use cosine decay down to min learning rate.
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, 
                 lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure: Callable|None=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None: continue
                state = self.state[p]
                t = state.get('t', 1)
                m = state.get('m', 0)
                v = state.get('v', 0)
                lrt = lr*((1-b2**t)**0.5)/(1-b1**t)
                p.data -= lr*weight_decay*p.data
                m = b1*m+(1-b1)*p.grad
                v = b2*v+(1-b2)*p.grad**2
                p.data -= lrt*m/(v**0.5+eps)
                state['t'] = t+1
                state['m'] = m
                state['v'] = v
        return loss

class AdamWOpt(torch.optim.Optimizer):
    def __init__(self, params,
                 lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            params, grads, ms, vs, steps = [], [], [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["t"] += 1
                params.append(p)
                grads.append(p.grad)
                ms.append(state["m"])
                vs.append(state["v"])
                steps.append(state["t"])

            if not params:
                continue

            # 假设同组参数 step 一致（常规情况成立），取一个算 bias correction
            t = steps[0]
            lrt = lr * (1 - b2 ** t) ** 0.5 / (1 - b1 ** t)

            # weight decay: p -= lr*wd*p   ->  p *= (1 - lr*wd)
            if weight_decay != 0:
                torch._foreach_mul_(params, 1 - lr * weight_decay)

            # m = b1*m + (1-b1)*g
            torch._foreach_mul_(ms, b1)
            torch._foreach_add_(ms, grads, alpha=1 - b1)

            # v = b2*v + (1-b2)*g^2
            torch._foreach_mul_(vs, b2)
            torch._foreach_addcmul_(vs, grads, grads, value=1 - b2)

            # p -= lrt * m / (sqrt(v) + eps)
            denom = torch._foreach_sqrt(vs)
            torch._foreach_add_(denom, eps)
            torch._foreach_addcdiv_(params, ms, denom, value=-lrt)

        return loss