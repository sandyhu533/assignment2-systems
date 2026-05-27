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
所有数据均表明NVTX时间与Python实际运行时间一致：
medum 512: 运行时间71.1；NVTX时间71.1ms
small 256: 运行时间16.13；NVTX时间16.1ms
small 512: 运行时间23.29；NVTX时间23.3ms
small 1024: 运行时间71.38；NVTX时间71.4ms

b. Forward 中最热的Kernel
ampere_sgemm_128x64_tn是Forward中最热的Kernel，在不同model_size/context_length中的占比和launch 次数如下：
medium 512: 56% 168 times
small 256: 72% 84 times
small 512: 56% 61 times
small 1024: 35% 21 times
说明该Kernel的占比随着context_length的增加而减少，但与model_size不成必然联系，主要原因是：
随着context_length的增加，attention中scores的计算和softmax会占用更多的GPU时间，这部分算子为ampere_sgemm_128x128_tn和element-wide居多，自然拉低了ampere_sgemm_128x64_tn的占比
另外这个Kernel在backward和optimize中显著占比降低，以medium 512为例，这个算子在optimize中占比0%，在backward中占比为5.9%，主要原因是：
optimizer step没有矩阵乘法，以elementwise操作为主
backward 一个 Linear 要 2 个 matmul（dX 和 dW），形状变了会选不同 sgemm tile，加上 softmax/RMSNorm/SiLU 的 backward 全是 elementwise，所以 128x64_tn 占比被稀释

c.
除了矩阵乘法以外，还看到大量的vectorized_elementwise_kernel，用于处理rms norm类算子中的乘法运算

d.adamw的kernel没有矩阵乘法，基本上全都是vectorized_elementwise_kernel

e. scale_dot_product_attention里面scores相关的矩阵乘法是ampere_sgemm_128x128_nn，而q k v proj的矩阵乘法是ampere_sgemm_128x64_tn；此外，scale_dot_product_attention中还有大量element_wise和vectorize_element_wise_kernel（占70%左右），以及少量reduce kernel（占10%左右）

通过观察发现scale_dot_product_attention中softmax耗时占整个scale_dot_product_attention的比例约50%，是性能瓶颈之一，而softmax的flops少到几乎可以忽略

# mixed_precision_accumulation
运行结果为：
float32+float32: tensor(10.0001) 
float16+float16: tensor(9.9531, dtype=torch.float16)
float32+float16: tensor(10.0021)
float32+float16->32: tensor(10.0021)
其中1、3、4的结果都基本符合预期，3、4的结果甚至完全一样，说明3也是python自动触发了16->32的转换；
fp32+fp32 也不是精确的 10（你的结果是 10.0001），原因是 0.01 在二进制浮点里本来就无法精确表示。所以 fp32 也有 round-off，只是非常小
2存在明显的“精度丢失”问题，缺少了0.05

```
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
```

# benchmarking_mixed_precision

a. 
>>>>>>>>>>>before autocasting>>>>>>>>>>>>
param dtype: torch.float32
input torch.float32
after fc1 torch.float32
after relu torch.float32
after ln torch.float32
after fc2 torch.float32
logits torch.float32
loss torch.float32
fc1.grad: torch.float32
fc2.grad: torch.float32
>>>>>>>>>>>after autocasting>>>>>>>>>>>>
param dtype: torch.float32
input torch.float32
after fc1 torch.float16
after relu torch.float16
after ln torch.float32
after fc2 torch.float16
logits torch.float16
loss torch.float32
fc1.grad: torch.float32
fc2.grad: torch.float32

b. 
FP32/FP16/BF16/TF32性质总结：
FP32全精度基线，慢但稳，每个值 4 bytes
FP16半精度，速度 2x，range 太窄，必须 GradScaler，训练易崩，每个值 2 bytes
BF16半精度，速度 2x，range 和 FP32 一样，精度更差但训练稳，不需 GradScaler，每个值 2 bytes，现代训练默认
TF32偷偷加速 FP32 matmul，存储不变，用户无感知

What parts of layer normalization are sensitive to mixed precision?

LN 中对低精度敏感的部分是 方差计算和归一化（Σ(x-μ)²、mean、rsqrt）。具体来说：

平方 (x²) 可能溢出 —— FP16 max 是 65504，深层网络的 activation 平方可能超过
大数相近相减 (x - μ) —— 灾难性相消，相对误差被放大
D 维 reduce 累加 —— 累积误差按 √D 增长，对 D=4096 的 transformer 显著
方差太小时 underflow —— FP16 下方差 < 6e-5 直接变成 0，导致 1/sqrt(0) 爆炸
rsqrt(small) 精度损失 —— 对小输入特别敏感

PyTorch 在 autocast 下把整个 LN upcast 到 FP32 计算，规避所有这些问题。

If we use BF16 instead of FP16, do we still need to treat LN differently?

理论上：基本不需要。 BF16 的 exponent range 与 FP32 一致，解决了 FP16 下最严重的 overflow（平方溢出）和 underflow（方差下溢）问题——这两个是 FP16 训练 LN 时最常导致 NaN/Inf 的元凶。
但实践上 PyTorch 仍然把 LN 用 FP32 算，原因：

BF16 的 mantissa 精度比 FP16 还差（7 bit vs 10 bit），相消问题和 reduce 累加误差反而更严重
LN 在整个 forward 里只占 1-3% 时间，FP32 计算开销可忽略
工业实践：fused LayerNorm kernel 普遍用 "BF16 storage + FP32 compute" 模式，两全其美

总结：BF16 让 LN 不再"会崩"，但还是"会偏几个 %"。为了那点稳定性，多花 1% 时间用 FP32 算 LN 是划算的。所以 PyTorch 默认依然把 LN 列入 BF16 黑名单。

c.
数据结果：
forward pass: 
small: 22.96 -> 19.25ms, 0.95 -> 0.81GB
medium: 70.97 -> 37.68ms, 2.15 -> 2.03GB
large: 164.94 -> 78.03ms, 4.48 -> 4.47GB

forward+backward pass:
small: 81.91 -> 58.43ms, 5.83GB -> 4.59GB
mediem: 234.55 -> 143.25ms, 15.07 -> 11.86GB
large: OOM -> 283.57ms, OOM -> 24.01GB

full pass:
small: 91.53 -> 74.06ms, 6.43 -> 5.11GB
medium: 265.58 -> 176.12ms, 17.23 -> 13.62GB
large: OOM -> OOM

Insights:
1. 普遍来说模型越大优化效果越明显，因为大模型更加compute-bound on matmul，amp主要优化matmul
2. GPU时间优化幅度forward > backward > optimize
optimize grad, opt status都是FP32，auto case无变化
forward比backward幅度大也是因为forward比backward更加compute-bound，backward每步还需要把梯度upcast，这部分也额外增加了成本
forward: 70.97 -> 37.68ms
backward: 163.58 -> 105.57ms
optimize: 31.03 -> 32.87ms
3. forward only Memory只有小幅下降，因为本身inference mode不用保存activation，模型参数量不受auto cast影响，小幅下降来自input output tensor和temp tensor；backward阶段Memry优化明显，而且参数量越大优化的越多，尤其是原本跑OOM的large模型在auto cast之后可以跑出来结果

NSys数据结果，以forward pass medium为例
**FP32** (主导)
- `ampere_sgemm_128x64_tn`: 56.0%, 39.57ms, 168 instances, **avg 235µs ± 151µs (StdDev 64%)** — 同一 kernel 覆盖多种 shape
- `ampere_sgemm_128x128_nn`: 5.3%, 3.76ms, 48 instances, avg 78µs ± 16µs

**AMP** (分散到多个专用 Tensor Core kernel)
- `s1688gemm_256x64_tn`: 14.4%, 5.02ms, 48 instances, avg 105µs ± 0.2µs
- `s1688gemm_128x64_tn`: 8.2%, 2.84ms, 96 instances, avg 30µs ± 0.3µs
- `s1688gemm_64x128_tn`: 7.4%, 2.59ms, 24 instances, avg 108µs ± 0.4µs
- `s1688gemm_128x128_nn`: 1.6%, 0.54ms, 24 instances, avg 23µs ± 0.2µs

Insights:
1. 矩阵乘相关算子的GPU时间占比显著下降（63%->35%），占比下降是因为GEMM显著加速而非 GEMM 没加速
2. 从原kernel变成了新的fp16系列kernel
3. 出现了更多的异构kernel类型，因为FP16使用的是Tensor Core，kernel catalog 更精细

为什么FP16更快：
1. **Raw compute**: BF16/FP16 Tensor Core = ~2× FP32 CUDA core (165 vs 83 TFLOPS on 4090)
2. **Memory bandwidth**: BF16/FP16 = half the bytes per element → halved memory traffic for memory-bound matmuls
3. **Specialized dispatch**: cuBLAS has 6 specialized Tensor Core variants vs 3 sgemm variants, giving better shape coverage

# memory_profiling

# gradient_checkpointing