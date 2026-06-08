import torch
import torch.distributed as dist
from typing import Any

class Zero1(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: torch.optim.Optimizer, **kwargs):
        # metadata of dist
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # all_params and param_owner
        self.all_params = []
        self.param_owner = []
                
        # optimizer cls and instance
        self.opt_cls = optimizer_cls
        self.opt_instance = None
        self.opt_kwargs = kwargs
        
        # super.init
        super().__init__(params, dict(kwargs))
    
    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        
        # get param group (filled default)
        group = self.param_groups[-1]
        
        # chunk params
        local_params = []
        for param in group["params"]:
            idx = len(self.all_params) % self.world_size
            self.all_params.append(param)
            self.param_owner.append(idx)
            if idx == self.rank: local_params.append(param)
        
        # add into local optimizer instance
        if local_params:
            other_params = {k:v for k, v in group.items() if k != "params"}
            if self.opt_instance is None:
                self.opt_instance = self.opt_cls([{"params": local_params, **other_params}], **self.opt_kwargs)
            else:
                self.opt_instance.add_param_group({"params": local_params, **other_params})
    
    def step(self, closure=None, **kwargs):
        # guard
        if closure is not None: return NotImplemented
        
        # update local
        self.opt_instance.step(closure, **kwargs)
        
        # broadcast
        for param, owner in zip(self.all_params, self.param_owner):
            dist.broadcast(param, src=owner)