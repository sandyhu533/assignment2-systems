# benchmarking_script
a. done
b. std small with warmup. tipical time:
- medium:
- forward 70ms
- foward+backward 232ms
- full 264ms
c. std high without warmup, much better with even 1-2 warmup
reason: gpu device warmup, gpu library lazy init causing first request longer than others

# nsys_profile

```
#全采样
uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx --cudabacktrace=all --python-backtrace=cuda -- python cs336_systems/benchmark.py

#只采样measure部分
uv run nsys profile \
  --trace=cuda,cudnn,cublas,osrt,nvtx \
  --pytorch=functions-trace,autograd-shapes-nvtx \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o my_profile \
  -- python cs336_systems/benchmark.py --size small --mode full --warmup 3 --steps 3
```

a. 
medium run in forward mode, pick only forward range NVTX Range Duration 72ms
vs medum run in foward mode, total duration 72ms
these two result match with each other

b.

forward:
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
56.3%	120.047 ms	504	238.188 μs	112.144 μs	103.905 μs	468.705 μs	148.509 μs	ampere_sgemm_128x64_tn

full:
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Category	Operation
15.6%	123.811 ms	504	245.657 μs	117.968 μs	105.665 μs	486.881 μs	153.756 μs	CUDA_KERNEL	ampere_sgemm_128x64_tn

c. for example:
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
5.0%	10.597 ms	72	147.176 μs	147.056 μs	144.864 μs	149.056 μs	842 ns	void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)2>>(int, T2, T3)

d.adamw的kernel没有矩阵乘法

e. 

# mixed_precision_accumulation

autocast 是怎么工作的?
torch.cuda.amp.autocast 并不是把所有东西都变成 fp16,它有一套规则:

weights 永远是 fp32 (master copy)
        │
        ↓ autocast cast 到 fp16 (临时副本)
        │
   forward 算到 logits (fp16)
        │
   loss (fp32, 自动 upcast,因为 softmax/log 是 numerically sensitive op)
        │
   backward → grad (fp16)
        │
   GradScaler.unscale_() → grad (fp32)   ← 注意这里
        │
   optimizer.step():  weight(fp32) += -lr * grad(fp32)
                      ↑ 这就是 fp32 accumulator

# benchmarking_mixed_precision

# memory_profiling

# gradient_checkpointing