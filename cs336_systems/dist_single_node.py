import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import timeit
import numpy as np
import json

def setup(rank, world_size, device, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if device == "cuda":
        torch.cuda.set_device(rank)

def emit_metrics(world_size, data_size, backend, device, warmup, steps, all_times):
    # all_times: list[world_size] of tensors shape [steps]
    mat = torch.stack(all_times).cpu().numpy()   # [world_size, steps]

    # per-step: 每步取所有 rank 的 max（collective 实际耗时由最慢 rank 决定）
    step_latency = mat.max(axis=0)               # [steps]
    avg_ms = step_latency.mean()

    stats = {
        "world_size": world_size,
        "backend": backend,
        "data_size": data_size,
        "device": device,
        "warmup": warmup,
        "steps": steps,
        # per-step 延迟分布（核心指标）
        "avg_ms": round(float(avg_ms), 2),
        "p50_ms": round(float(np.percentile(step_latency, 50)), 2),
        "p99_ms": round(float(np.percentile(step_latency, 99)), 2),
        "max_ms": round(float(step_latency.max()), 2),
        "std_ms": round(float(step_latency.std()), 2),
        # per-rank 均值（straggler 检测）
        "per_rank_avg_ms": [round(float(x), 2) for x in mat.mean(axis=1)],
        "step_latency_ms": [round(float(x), 2) for x in step_latency]
    }
    print(json.dumps(stats, indent=2))
      
def distributed_demo(rank, world_size, data_size, device, backend, warmup, steps):
    setup(rank, world_size, device, backend)
    
    for _ in range(warmup):
        data = torch.rand(data_size, dtype=torch.float32, device=device)
        dist.all_reduce(data, async_op=False)
        if device == "cuda":
            torch.cuda.synchronize()
            
    times = []
    for _ in range(steps):
        data = torch.rand(data_size, dtype=torch.float32, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = timeit.default_timer()
        dist.all_reduce(data, async_op=False, op=dist.ReduceOp.SUM)
        if device == "cuda":
            torch.cuda.synchronize()
        used_time = round((timeit.default_timer()-t0)*1e3,2)
        times.append(used_time)
    
    local_times = torch.tensor(times, dtype=torch.float32, device=device)  # shape [steps]
    if rank == 0:
        all_times = [torch.zeros(len(times), dtype=torch.float32, device=device)
                 for _ in range(world_size)]
    else:
        all_times = None
    dist.gather(local_times, all_times, dst=0)
    if rank == 0:
        emit_metrics(world_size, data_size, backend, device, warmup, steps, all_times)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--data_size_mb", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    world_size = args.world_size
    data_size = (args.data_size_mb * 262144, )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = "gloo" if device == "cpu" else "nccl"
    print(f'start runing with config {world_size=} {data_size=}')
    
    mp.spawn(fn=distributed_demo, args=(world_size, data_size, device, backend, args.warmup, args.steps), nprocs=world_size, join=True)
        
if __name__ == "__main__":
    main()