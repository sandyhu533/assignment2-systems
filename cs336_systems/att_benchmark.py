import argparse
import torch
import cs336_basics
from cs336_basics import model, nn_utils, optimizer, data
import timeit
import numpy as np
import json, sys
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext

cs336_basics.model.scaled_dot_product_attention = cs336_basics.model.annotated_scaled_dot_product_attention

def emit_result(args, stats):
    """stats: dict with mean_ms, std_ms, median_ms, p95_ms"""
    payload = {
        "mode":   args.mode,
        "size":   args.size,
        "warmup": args.warmup,
        "steps":  args.steps,
        **stats,
    }
    print("RESULT_JSON " + json.dumps(payload), file=sys.stdout, flush=True)
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--context_length", type=int,default=256)
    parser.add_argument("--d_model", type=int,default=16)
    parser.add_argument("mode", type=str, default="forward", choices=["forward", "forward_backward"])
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--mem_snapshot", type=str)

    
    args = parser.parse_args()
    context_length = args.context_length
    num_heads = 1
    d_k = args.d_model
    batch_size=8
    device="cuda"
    
    autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if args.use_amp else nullcontext()
    fwd_ctx = torch.inference_mode() if args.mode == "forward" else nullcontext()
    
    def model_step():
        with fwd_ctx:
            q = torch.randint(0, 100, (batch_size, num_heads, context_length, d_k))
            k = torch.randint(0, 100, (batch_size, num_heads, context_length, d_k))
            v = torch.randint(0, 100, (batch_size, num_heads, context_length, d_k))
            mask = torch.trill(torch.ones((context_length, context_length), dtype=torch.bool))
            att = model.annotated_scaled_dot_product_attention(
                q, k, v, mask
            )
            torch.cuda.synchronize()
        if args.mode.contains("backward"):
            att.backward()
            torch.cuda.synchronize()
    
    for _ in range(args.warmup):
        model_step()
    
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    torch.cuda.cudart().cudaProfilerStart()
    times = []
    for i in range(args.steps):
        with nvtx.range("one_pass"), autocast_ctx:
            t0 = timeit.default_timer()
            model_step()
            times.append(timeit.default_timer()-t0)
        print(f"step {i} time={round(times[-1],2)}ms mem={round(torch.cuda.max_memory_allocated(device)/1e9, 2)}GB")
    torch.cuda.cudart().cudaProfilerStop()
    
    suffix = "-amp" if args.amp else ""
    torch.cuda.memory._dump_snapshot(args.mem_snapshot)
    torch.cuda.memory._record_memory_history(enabled=None)
    
    alloc_mem = torch.cuda.max_memory_allocated(device)
    reserve_mem= torch.cuda.max_memory_reserved(device)
    
    time_ms = np.array(times) * 1e3
    stats = {
        "mean_ms": time_ms.mean(),
        "std_ms": time_ms.std(),
        "median_ms": np.median(time_ms),
        "p95_ms": np.percentile(time_ms, 95),
        "peak_mem_gb": reserve_mem/1e9,
        "peak_alloc_mem_gb": alloc_mem/1e9,
    }
    for k, v in stats.items():
        stats[k] = round(v, 2)
    emit_result(args, stats)

main()