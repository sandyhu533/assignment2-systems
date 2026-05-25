import argparse
import torch
import cs336_basics
from cs336_basics import model, nn_utils, optimizer, data
import timeit
import numpy as np
import json, sys
import torch.cuda.nvtx as nvtx

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
    parser.add_argument("--size", type=str, default="small", choices=["small", "medium", "large", "xl", "10B"],)
    parser.add_argument("--mode", type=str, default="full", choices=["forward", "forward_backward", "full"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--context_length", type=int,default=512)
    
    args = parser.parse_args()
    context_length = args.context_length
    vocab_size=10000
    batch_size=4
    device="cuda"

    model_config = {
        "small": {
            "d_model": 768,
            "d_ff": 3072,
            "num_layers": 12,
            "num_heads": 12
        },
        "medium": {
            "d_model": 1024,
            "d_ff": 4096,
            "num_layers": 24,
            "num_heads": 16
        },
        "large": {
            "d_model": 1280,
            "d_ff": 5120,
            "num_layers": 36,
            "num_heads": 20
        },
        "xl": {
            "d_model": 2560,
            "d_ff": 10240,
            "num_layers": 32,
            "num_heads": 32
        },
        "10B": {
            "d_model": 4608,
            "d_ff": 12288,
            "num_layers": 50,
            "num_heads": 36
        }
    }

    config = model_config[args.size]
    need_optimize = args.mode == "full"
        
    random_dataset = np.random.randint(0, vocab_size, size=100_000, dtype=np.int32)
    
    rope = model.RotaryPositionalEmbedding(10000, context_length, config["d_model"]//config["num_heads"],device)
    m = model.TransformerLM(
        vocab_size,
        config["num_layers"],
        config["d_model"],
        config["d_ff"],
        context_length,
        config["num_heads"],
        rope=rope,
        device=device
    )
    opt = optimizer.AdamW(m.parameters())
    
    # warm up
    nvtx.range_push("warmup")
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        x, y = data.get_batch(random_dataset, batch_size, context_length, device)
        if args.mode == "forward":
            with torch.inference_mode():
                outputs = m(x)
        else:
            outputs = m(x)
            loss = nn_utils.cross_entropy(outputs, y)
            loss.backward()
            if need_optimize:
                opt.step()
            opt.zero_grad()
        torch.cuda.synchronize()
    nvtx.range_pop()
    
    # nvtx.range_push("measure")
    torch.cuda.cudart().cudaProfilerStart()
    times = []
    alloc_mem = []
    reserve_mem = []
    torch.cuda.synchronize()
    for _ in range(args.steps):
        with nvtx.range("one_pass"):
            x, y = data.get_batch(random_dataset, batch_size, context_length, device)
            t0 = timeit.default_timer()
            if args.mode == "forward":
                with nvtx.range("forward"):
                    with torch.inference_mode():
                        outputs = m(x)
                    torch.cuda.synchronize()
            else:
                with nvtx.range("forward"):
                    outputs = m(x)
                    loss = nn_utils.cross_entropy(outputs, y)
                    torch.cuda.synchronize()
                with nvtx.range("loss"):
                    loss = nn_utils.cross_entropy(outputs, y)
                    torch.cuda.synchronize()
                with nvtx.range("backward"):
                    loss.backward()
                    torch.cuda.synchronize()
                if need_optimize:
                    with nvtx.range("optimize"):
                        opt.step()
                        opt.zero_grad()
                        torch.cuda.synchronize()
            times.append(timeit.default_timer()-t0)
    # nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    
    alloc_mem = torch.cuda.max_memory_allocated(device)
    reserve_mem= torch.cuda.max_memory_reserved(device)
    
    time_ms = np.array(times) * 1e3
    stats = {
        "mean_ms": time_ms.mean(),
        "std_ms": time_ms.std(),
        "median_ms": np.median(time_ms),
        "p95_ms": np.percentile(time_ms, 95),
        "peak_reserve_mem_gb": np.max(np.array(reserve_mem))/1e9,
        "peak_alloc_mem_gb": np.max(np.array(alloc_mem))/1e9,
    }
    for k, v in stats.items():
        stats[k] = round(v, 2)
    emit_result(args, stats)

main()