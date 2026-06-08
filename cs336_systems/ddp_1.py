import torch
import torch.distributed as dist

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []
        
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        self._register_hooks()
    
    def _register_hooks(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._gradient_hook)
    
    def _gradient_hook(self, parameter: torch.Tensor):
        if parameter.grad is not None:
            parameter.grad /= self.world_size
            
            handle = dist.all_reduce(parameter.grad, async_op=True)
            self.handles.append(handle)
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
    
    