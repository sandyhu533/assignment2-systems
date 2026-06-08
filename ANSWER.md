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

> > > > > > > > > > > before autocasting>>>>>>>>>>>>
> > > > > > > > > > > param dtype: torch.float32
> > > > > > > > > > > input torch.float32
> > > > > > > > > > > after fc1 torch.float32
> > > > > > > > > > > after relu torch.float32
> > > > > > > > > > > after ln torch.float32
> > > > > > > > > > > after fc2 torch.float32
> > > > > > > > > > > logits torch.float32
> > > > > > > > > > > loss torch.float32
> > > > > > > > > > > fc1.grad: torch.float32
> > > > > > > > > > > fc2.grad: torch.float32
> > > > > > > > > > >
> > > > > > > > > > > > > > > > > > > > > after autocasting>>>>>>>>>>>>
> > > > > > > > > > > > > > > > > > > > > param dtype: torch.float32
> > > > > > > > > > > > > > > > > > > > > input torch.float32
> > > > > > > > > > > > > > > > > > > > > after fc1 torch.float16
> > > > > > > > > > > > > > > > > > > > > after relu torch.float16
> > > > > > > > > > > > > > > > > > > > > after ln torch.float32
> > > > > > > > > > > > > > > > > > > > > after fc2 torch.float16
> > > > > > > > > > > > > > > > > > > > > logits torch.float16
> > > > > > > > > > > > > > > > > > > > > loss torch.float32
> > > > > > > > > > > > > > > > > > > > > fc1.grad: torch.float32
> > > > > > > > > > > > > > > > > > > > > fc2.grad: torch.float32

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

a. memory lifecycle分析：

forward pass with xl model size

memory timeline分为几个部分：

- 模型参数部分占大多数，不随着时间变化
- activation部分占小数
  - 会随着时间变化，跟随transformer的周期运行，过程中会产生局部变量但在函数生命周期结束之后会被销毁
  - 由于开启了inference mode，不保存activation  
  最高峰在Attention scores softmax的位置

full pass with medium model size

最高峰在forward pass刚算完activation的时候

整体的时间轴分为几个阶段：

- 基线是有模型参数+state.m和v
- forward pass：累计activations
- backward pass：计算p.grad，同时释放activations
- optimizer.step()：根据p.grad更新p.data和state.m和v
- optimizer.zero_grad()：释放p.grad

b. memory usage随context_length的变化（forward+full）

forward pass with small model size (conetxt_length 256/512/1024)
峰值显存和最大内存分配和context_length呈正相关关系
峰值显存：562MB -> 705MB -> 1.2GB
最大内存分配：12MB -> 48MB -> 192MB
基线（模型参数）为580MB，在context_length达到1024的时候，明显看出activation和模型参数对显存的占用量量级已经很接近了

full pass with small model size (conetxt_length 256/512/1024)
峰值显存和context_length呈正相关关系
峰值显存：2.7GB -> 4.4GB -> 9.5GB
基线（模型参数+state）为2.2GB，峰值显存在context_length为1024时远超这个数

c. mixed-precision对memory的影响（forward+full）

forward pass with xl model size
- 整体看起来优化不大，主要影响attn和ffn的activation
- attn分配内存最多的部分是softmax计算，这块由于softmax操作在amp的黑名单所以也没受到影响
- 由于开启了amp会多一些精度转换带来的临时内存申请

full pass with medium model size
- 基线（模型参数和state）没有产生变化
- 峰值有明显下降，从15.2GB下降到12.3GB，反映出activation cast成fp16的影响

d. xl model transformer residual stream大小

以amp的结果为例：
residual stream的大小是20.0MiB，和理论计算值一致
- batch_size*context_length*d_model*4=2560*4*512*4/1024/1024=20MB
attn_output和ffn_output的大小都是10.0MB
- 因为这些层属于“白名单”被amp转成了fp16，所以大小是residual stream的一半
ln_output的大小则是20.0MB
- ln属于“黑名单”，大小等于residual stream

e. 分析xl model forward pass的最大内存分配
What is the size of the largest allocations  
shown? Looking through the stack trace, can you tell where those allocations come from?
最大的memory allocation是128MB，对应attn的softmax阶段

f. Nsight Systems memory profiling

以meduim 512 context_length模型的full pass的无amp版本内存分配为例，其参数为：
# "medium": {
#             "d_model": 1024,
#             "d_ff": 4096,
#             "num_layers": 24,
#             "num_heads": 16
#         },
# context_lengt=512, back_size=4
可以看到每个attention block大概是这样的activation结构：
5*8MB + 3*64MB + 5*8MB + 5*32MB
其中比较明确的是
- 64MB = batch_size*context_length*context_length*num_heads*4
- 8MB = batch_size*context_length*d_model*4
- 32MB = batch_size*context_length*d_ff*4


第一个5*8MB是：
1. pynumber_add - residual stream输入（上一层的ffn output或者embedding lookup结果）
2. pynumber_truedivice - rmsnorm的r/rms
3. pynumber_mult - rmsnorm的x_normed*weight
4. Q proj
5. K proj

3*64MB是：
1. scores = q*k
2. mask(scores) # 可以通过inplace操作优化掉这一步
3. softmax(scores)

第二个5*8MB是：
1. attn_output = softmax(scores)*v
2. output proj
3. 把attn的结果累加到residual stream
4. pynumber_truedivice - rmsnorm的r/rms
5. pynumber_mult - rmsnorm的x_normed*weight

5*32MB是：
1. w1x
2. w2x
3. sigmoid(w1x)
4. sigmoid(w1x)*w1x
5. sigmoid(w1x)*w1x*w2x

# gradient_checkpointing

a. 假设不考虑计算成本，memory降低最多的策略是只保留输入的embedding output，每次backward都从头开始计算算到最顶层正在做backward的layer，直到把所有的layer的梯度都算完。
这样的显存消耗是O(1)，计算消耗是O(N^2)

b. 假设最多只能做一次重计算，最优策略是保存residual stream，即进入ln1+attn之前的x和进入ln2+ffn之前的x，这种策略的显存消耗是2*O(N) + max(attn_activation, ffn_activation)，计算消耗是O(N)

题目的本意是不考虑常数项，计算应该把Transformer Block分成几段。假设总共有N个Block，要分成K端，保存的checkpoint=K*a，峰值activation=N/K*b，总消耗等于K*a + N/K*b，求导得出最小值的K= (N*b/a)^(1/2)，在a==b的时候K=N^(1/2)

但实际上由于activation的常数项远大于residual stream的常数项，所以实际的最优策略一般是K=N即每个transformer block都保存一个checkpoint

至于为什么不考虑最开始提出的这种方案是因为工程实现复杂，roi比较低

# pytorch_attention
benchmark script in cs336_systems/att_benchmark.py

| d_model | ctx | forward_ms | backward_ms | peak_mem_gb | peak_alloc_mem_gb |
|---|---|---|---|---|---|
| 16 | 256 | 1.62 | 2.60 | 0.04 | 0.03 |
| 16 | 1024 | 0.89 | 2.55 | 0.36 | 0.22 |
| 16 | 4096 | 8.06 | 22.28 | 5.43 | 3.27 |
| 16 | 8192 | 31.70 | 86.77 | 21.61 | 13.00 |
| 16 | 16384 | OOM | OOM | — | — |
| 32 | 256 | 0.86 | 2.42 | 0.05 | 0.03 |
| 32 | 1024 | 0.89 | 2.47 | 0.37 | 0.23 |
| 32 | 4096 | 8.08 | 22.38 | 5.96 | 3.29 |
| 32 | 8192 | 31.69 | 87.42 | 23.76 | 13.04 |
| 32 | 16384 | OOM | OOM | — | — |
| 64 | 256 | 0.85 | 2.48 | 0.05 | 0.03 |
| 64 | 1024 | 0.89 | 2.48 | 0.39 | 0.24 |
| 64 | 4096 | 8.14 | 22.35 | 5.98 | 3.32 |
| 64 | 8192 | 31.89 | 86.97 | 23.79 | 13.10 |
| 64 | 16384 | OOM | OOM | — | — |
| 128 | 256 | 0.84 | 2.52 | 0.05 | 0.04 |
| 128 | 1024 | 0.89 | 2.49 | 0.41 | 0.25 |
| 128 | 4096 | 8.43 | 22.91 | 6.57 | 3.39 |
| 128 | 8192 | 32.99 | 88.99 | 19.59 | 13.24 |
| 128 | 16384 | OOM | OOM | — | — |

## 最小OOM参数量计算
activation for d_model=16, ctx=16384
B=8, L=16384, D=16, num_heads=1
= 3BLD+3BLL*num_heads+2BL*num_heads+LL
= 25,165,824 + 25,769,803,776 + 1,048,576 + 268,435,456 = 26,064,453,632 bytes ≈ 24.27 GiB
GPU 0 has a total capacity of 23.54 GiB，所以必然会OOM

## How does the memory saved for
backward change with the sequence length
与sequence length的平方成正比

## What would you do to eliminate this memory
cost
主导项来自softmax和attn@v，即计算的中间结果，可以尽量减少中间结果的存储，用checkpointing方法在backward的时候重新做计算

# torch_compile

## attention
- compile之后forward和backward_ms有大幅下降，peak_mem_gb也有大幅下降
- compile之后peak_mem_gb和peak_alloc_mem_gb更加接近，原因是 compile 后峰值内存大幅减小，碎片（reserved 但未 alloc 的部分）随之减少

ctx=4096, d_model=64, fp32, forward_backward

| 配置 | compile | forward_ms | backward_ms | peak_mem_gb | peak_alloc_mem_gb |
|---|---|---|---|---|---|
| Annotated | ✗ | 8.86 | 22.43 | 5.98 | 3.32 |
| Annotated | ✓ | 4.79 | 7.04 | 2.28 | 2.25 |
| Not annotated | ✗ | 8.16 | 22.33 | 5.98 | 3.32 |
| Not annotated | ✓ | 2.77 | 6.24 | 2.28 | 2.25 |

## full model

forward_backward，fp32，warmup=1, steps=1

| size | compile | mean_ms | peak_mem_gb | peak_alloc_mem_gb |
|---|---|---|---|---|
| medium | ✗ | 241.92 | 13.80 | 12.95 |
| medium | ✓ | 162.96 | 9.78 | 9.32 |
| large | ✗ | OOM | — | — |
| large | ✓ | 386.16 | 18.17 | 17.97 |

# flash_forward

a. skip to save time, 直接写triton吧
b. trition implementation done

# flash_backward

# flash_benchmarking

# fsdp_accounting
Param = 4N -> 4N/G
Grad = 4N -> 4N/G
Opt = 8N -> 8N/G

Param = 2N -> 2N/G
Grad = 2N -> 2N/G
Opt = 12N -> 12N/G

# alternate_ring_all_reduce
t = (N-1)*S/W

# data_parallel_calcs

x1, x2, z (B, DF)
y (B, D)

a) FLOPS = 

24: 2*B*D*DF
25: skip
26: skip
27: 2*2*B*DF*D
28, 29, 30: 2*B*D*DF

Final flops = 12*B*D*DF/Ndp

time = 12*B*D*DF/(Ndp*C)

b)

普通DP需要All reduce，因此需要两次通信

S=3*D*DF*2*2 (FP16, All reduce)
time=(N-1)*S/(N*W)
communication time = (N-1)/N * (12*D*DF)/W

c)

bottlenecked:
Ndp = BW/C

# fsdp_calcs

a)

forward flops = 6*B*D*DF/N
backward flops = 12*B*D*DF/N

b)

each all_gather or reduce-scatter: (N-1)/N * (6*D*DF)/W

forward = (N-1)/N * (6*D*DF)/W
backward = 2 * (N-1)/N * (6*D*DF)/W

c)

forward bottleneck: N=BW/C+1
backward bottleneck: N=BW/C+1

# tp_calcs

a) dx = AllReduce_i(dx_partial^(i))

b)
forward flops = 6*B*D*DF/N
backward flops = 12*B*D*DF/N

c)
S = B*D*2*2 (FP16, All-reduce)
forward & backward time
=(N-1)*S/(N*W)
= (N-1)*2*B*D/(N*W)

d)

bottleneck 
forward N = 3*W*Df/2*C + 1
backward N = 3*W*Df/C + 1