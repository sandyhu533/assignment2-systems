import argparse
import torch
import cs336_basics
from cs336_basics import model
import timeit
import numpy as np
import json, sys
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext

def emit_result(args, stats):
    """stats: dict with mean_ms, std_ms, median_ms, p95_ms"""
    payload = {
        **vars(args),
        **stats,
    }
    print("RESULT_JSON " + json.dumps(payload), file=sys.stdout, flush=True)
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--context_length", type=int,default=256)
    parser.add_argument("--d_model", type=int,default=16)
    parser.add_argument("--mode", type=str, default="forward", choices=["forward", "forward_backward"])
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--mem_snapshot", type=str, default="test.pickle")
    parser.add_argument("--compile", action="store_true")
    
    args = parser.parse_args()
    context_length = args.context_length
    num_heads = 1
    d_k = args.d_model
    batch_size=8
    device="cuda"
    
    autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if args.use_amp else nullcontext()
    fwd_ctx = torch.inference_mode() if args.mode == "forward" else nullcontext()
    has_backward = args.mode == "forward_backward"
    mask = torch.tril(torch.ones((context_length, context_length), device=device, dtype=torch.bool))
    q = torch.rand((batch_size, num_heads, context_length, d_k), device=device, requires_grad=True)
    k = torch.rand((batch_size, num_heads, context_length, d_k), device=device, requires_grad=True)
    v = torch.rand((batch_size, num_heads, context_length, d_k), device=device, requires_grad=True)
    
    attention_layer = model.scaled_dot_product_attention
    if args.compile:
        attention_layer = torch.compile(attention_layer)
    forward_times = []
    backward_times = []
    def forward_step(warmup, idx):
        t0 = timeit.default_timer()
        
        with nvtx.range("one_pass"), autocast_ctx:
            with fwd_ctx:
                att = attention_layer(
                    q, k, v, mask
                )
                torch.cuda.synchronize()
            forward_time = round((timeit.default_timer()-t0)*1e3,2)
            if not warmup:
                forward_times.append(forward_time)
            print(f'{warmup=} {idx=} {forward_time=}ms')
        return att
    
    def backward_step(att, warmup, idx):
        t1 = timeit.default_timer()
        if has_backward:
            with nvtx.range("one_pass"):
                att.sum().backward(retain_graph=True)
                torch.cuda.synchronize()
        backward_time = round((timeit.default_timer()-t1)*1e3,2)
        if not warmup:
            backward_times.append(backward_time)
        print(f'{warmup=} {idx=} {backward_time=}ms')

    att = None
    for i in range(args.warmup):
        att = forward_step(True, i)
    for i in range(args.warmup):
        backward_step(att, True, i)
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    torch.cuda.cudart().cudaProfilerStart()
    att = None
    for i in range(args.steps):
        att = forward_step(False, i)
    alloc_mem = torch.cuda.max_memory_allocated(device)
    reserve_mem= torch.cuda.max_memory_reserved(device)
    for i in range(args.steps):
        backward_step(att, False, i)
    torch.cuda.cudart().cudaProfilerStop()
    
    torch.cuda.memory._dump_snapshot(args.mem_snapshot)
    torch.cuda.memory._record_memory_history(enabled=None)
    
    forward_ms = np.array(forward_times)
    backward_ms = np.array(backward_times)
    stats = {
        "forward_ms": forward_ms.mean(),
        "backward_ms": backward_ms.mean(),
        "peak_mem_gb": reserve_mem/1e9,
        "peak_alloc_mem_gb": alloc_mem/1e9,
    }
    for k, v in stats.items():
        stats[k] = round(v, 2)
    emit_result(args, stats)

main()