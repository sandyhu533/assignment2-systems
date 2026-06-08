import torch
import torch.distributed as dist
from typing import Any

class NaiveDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for p in self.module.parameters():
            if p.requires_grad and p.grad is not None:
                dist.all_reduce(p.grad, dist.ReduceOp.AVG)

class FlattenDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        params = [p for p in self.module.parameters() if p.requires_grad]
        grads = [p.grad for p in params]
        flatten = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flatten, dist.ReduceOp.AVG)
        grads = torch._utils._unflatten_dense_tensors(flatten, grads)
        for grad, param in zip(grads, params):
            param.grad.copy_(grad)

class OverlapDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        for p in self.parameters():
            dist.broadcast(p.data, src=0)
        
        self.handles = []
        for p in self.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._grad_hook)
    
    def _grad_hook(self, param: torch.Tensor):
        if param.grad is not None:
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
            self.handles.append(handle)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        
        self.handles.clear()

