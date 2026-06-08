from typing import Any, Type
import torch
import torch.distributed as dist
from torch.optim import Optimizer

class NaiveDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

        for p in self.module.parameters():
            dist.broadcast(p, src=0)
    
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
            dist.broadcast(p, src=0)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        params = [p for p in self.module.parameters() if p.requires_grad and p.grad is not None]
        grads = [p.grad for p in params]
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat, dist.ReduceOp.AVG)
        grads = torch._utils._unflatten_dense_tensors(flat, grads)
        for grad, param in zip(grads, params):
            param.grad.copy_(grad)     

class OverlapDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        
        # boardcast initial params
        for p in self.module.parameters():
            dist.broadcast(p, src=0)
        
        # register hook
        self.handles = []
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._grad_hook)
        
    
    def _grad_hook(self, param: torch.Tensor):
        if param.grad is not None:
            handle = dist.all_reduce(param.grad, dist.ReduceOp.AVG, async_op=True)
            self.handles.append(handle)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

class Zero1(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: torch.optim.Optimizer, **kwargs):
        # dist meta data
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # all_param, param_owner
        self.all_param = []
        self.param_owner = []
        
        # opt clss lazy init
        self.opt_instance = None
        self.opt_cls = optimizer_cls
        self.opt_kwargs = kwargs
        
        # call super
        super().__init__(params, dict(kwargs))
        
        print(f"[rank {self.rank}] all_params[0] norm = {self.all_param[0].data.norm()} param_owner[:world_size] = {self.param_owner[:2*self.world_size]}")

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        
        # get param group from super
        group = self.param_groups[-1]
        
        # chunk param
        local_param = []
        for param in group["params"]:
            idx = len(self.all_param)
            owner = idx % self.world_size
            self.all_param.append(param)
            self.param_owner.append(owner)
            if owner == self.rank: local_param.append(param)
        
        # add pg into opt
        if local_param:
            other_param = {k:v for k,v in group.items() if k != "params"}
            if self.opt_instance is None:
                self.opt_instance = self.opt_cls([{"params": local_param, **other_param}], **self.opt_kwargs)
            else:
                self.opt_instance.add_param_group({"params": local_param, **other_param})
    
    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        if closure is not None: return NotImplemented
        
        # update local param
        self.opt_instance.step(closure, **kwargs)
        
        # broadcast chunked param
        for p, owner in zip(self.all_param, self.param_owner):
            dist.broadcast(p.data, src=owner)
        
        return None
        