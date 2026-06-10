import torch
import torch.distributed as dist
from typing import Any
import timeit

class NaiveDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.comm_times = []
        
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        torch.cuda.synchronize()
        t1 = timeit.default_timer()
        data_bytes = 0
        for p in self.module.parameters():
            if p.requires_grad and p.grad is not None:
                dist.all_reduce(p.grad, dist.ReduceOp.AVG)
                data_bytes += p.grad.numel() * p.grad.element_size()
        torch.cuda.synchronize()
        comm_time = timeit.default_timer() - t1
        
        backend = dist.get_backend()
        is_nccl = backend == "nccl"
        world_size = dist.get_world_size()
        transferred_bytes = 2 * (world_size - 1) / world_size * data_bytes
        bw_gbps = transferred_bytes / comm_time / 1e9 if comm_time > 0 else 0.0

        rank = dist.get_rank()
        print(f'rank{rank} backend={backend} is_nccl={is_nccl} '
              f'data={round(data_bytes/1e6,2)}MB '
              f'transferred={round(transferred_bytes/1e6,2)}MB '
              f'comm_time={round(comm_time*1e3,2)}ms '
              f'bw={round(bw_gbps,2)}GB/s')

        self.comm_times.append(comm_time)
        
class ChunkedFlattenDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_bytes=25 * 1024**2):
        super().__init__()
        self.module = module
        self.bucket_bytes = bucket_bytes   # 每组上限字节，控制显存峰值；调小省显存、调大提带宽
        self.comm_times = []
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    @staticmethod
    def _chunk_by_bytes(params, max_bytes):
        """按累计梯度字节数切组；单参数超过 max_bytes 时自成一组。"""
        groups, cur, cur_bytes = [], [], 0
        for p in params:
            b = p.grad.numel() * p.grad.element_size()
            if cur and cur_bytes + b > max_bytes:
                groups.append(cur)
                cur, cur_bytes = [], 0
            cur.append(p)
            cur_bytes += b
        if cur:
            groups.append(cur)
        return groups

    def finish_gradient_synchronization(self):
        torch.cuda.synchronize()
        t1 = timeit.default_timer()

        params = [p for p in self.module.parameters()
                  if p.requires_grad and p.grad is not None]
        chunks = self._chunk_by_bytes(params, self.bucket_bytes)

        data_bytes = 0
        for group in chunks:
            grads = [p.grad for p in group]
            flatten = torch._utils._flatten_dense_tensors(grads)
            data_bytes += flatten.numel() * flatten.element_size()
            dist.all_reduce(flatten, dist.ReduceOp.AVG)
            unflat = torch._utils._unflatten_dense_tensors(flatten, grads)
            for g, p in zip(unflat, group):
                p.grad.copy_(g)
            del flatten, unflat, grads   # 释放本组 flat buffer，下一组复用

        torch.cuda.synchronize()
        comm_time = timeit.default_timer() - t1

        backend = dist.get_backend()
        is_nccl = backend == "nccl"
        world_size = dist.get_world_size()
        transferred_bytes = 2 * (world_size - 1) / world_size * data_bytes
        bw_gbps = transferred_bytes / comm_time / 1e9 if comm_time > 0 else 0.0

        rank = dist.get_rank()
        print(f'rank{rank} backend={backend} is_nccl={is_nccl} '
              f'nbuckets={len(chunks)} '
              f'data={round(data_bytes/1e6,2)}MB '
              f'transferred={round(transferred_bytes/1e6,2)}MB '
              f'comm_time={round(comm_time*1e3,2)}ms '
              f'bw={round(bw_gbps,2)}GB/s')

        self.comm_times.append(comm_time)
        
class FlattenDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.comm_times = []
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        torch.cuda.synchronize()
        t1 = timeit.default_timer()
        params = [p for p in self.module.parameters() if p.requires_grad]
        grads = [p.grad for p in params]
        flatten = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flatten, dist.ReduceOp.AVG)
        grads = torch._utils._unflatten_dense_tensors(flatten, grads)
        for grad, param in zip(grads, params):
            param.grad.copy_(grad)
        torch.cuda.synchronize()
        
        comm_time = timeit.default_timer() - t1
        
        # 后端是否 nccl
        backend = dist.get_backend()
        is_nccl = backend == "nccl"

        # 数据大小（字节）：flatten 后的总字节数
        data_bytes = flatten.numel() * flatten.element_size()

        # AllReduce 实际传输量 = 2*(N-1)/N * data_bytes
        world_size = dist.get_world_size()
        transferred_bytes = 2 * (world_size - 1) / world_size * data_bytes

        # 带宽 GB/s = transferred_bytes / comm_time / 1e9
        bw_gbps = transferred_bytes / comm_time / 1e9 if comm_time > 0 else 0.0

        rank = dist.get_rank()
        print(f'rank{rank} backend={backend} is_nccl={is_nccl} '
              f'data={round(data_bytes/1e6,2)}MB '
              f'transferred={round(transferred_bytes/1e6,2)}MB '
              f'comm_time={round(comm_time*1e3,2)}ms '
              f'bw={round(bw_gbps,2)}GB/s')

        self.comm_times.append(comm_time)
                

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

