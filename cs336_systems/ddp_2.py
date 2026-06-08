import torch
import torch.distributed as dist

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        
        # init
        for p in self.parameters():
            dist.broadcast(p, src=0)
        
        self.handles = []
        # register hook
        for p in self.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._grad_hook)
        
    def _grad_hook(self, param: torch.Tensor):
        if param.grad is not None:
            # sync
            # dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            # async
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
            self.handles.append(handle)
            
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()