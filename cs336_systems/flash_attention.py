import torch
import triton
import triton.language as tl

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    bid = tl.program_id(0)
    qid = tl.program_id(1)
    
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + bid*stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(qid*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + bid*stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + bid*stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + bid*stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(qid*Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + bid*stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(qid*Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    
    k_offsets = tl.arange(0, K_TILE_SIZE)
    q_offsets = tl.arange(0, Q_TILE_SIZE) + qid*Q_TILE_SIZE
    
    q_vals = tl.load(Q_block_ptr, boundary_check=(0,1))
    acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    g_max = tl.full((Q_TILE_SIZE,), value=-float('inf'),dtype=tl.float32)
    g_sum = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    for k in range(0, N_KEYS, K_TILE_SIZE):
        # get k, v value
        k_vals = tl.load(K_block_ptr, boundary_check=(0,1))
        v_vals = tl.load(V_block_ptr, boundary_check=(0,1))
        
        # q@k/d**0.5
        qk = tl.dot(q_vals, tl.trans(k_vals))*scale
        if is_causal:
            casual_mask = q_offsets[:,None] >= (k+k_offsets)[None,:]
            qk = tl.where(casual_mask, qk, -1e6)
        qk_mask = k_offsets < N_KEYS-k
        qk = tl.where(qk_mask[None, :], qk, -float('inf'))

        # cur_max, new_g_max
        cur_max = tl.max(qk, -1)
        new_g_max = tl.maximum(cur_max, g_max)
        alpha = tl.exp(g_max-new_g_max)
        
        # apply alpha
        g_sum *= alpha
        acc = acc * alpha[:,None]

        # cur_qk
        cur_qk = tl.exp(qk - new_g_max[:,None])

        # update g_sum, g_max, acc
        g_max = new_g_max
        g_sum += tl.sum(cur_qk, -1)
        acc = tl.dot(cur_qk, v_vals, acc)
        
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE,0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE,0))
        
    
    tl.store(O_block_ptr, acc/g_sum[:,None], boundary_check=(0,1))
    tl.store(L_block_ptr, g_max + tl.log(g_sum), boundary_check=(0,))
    

class FlashAttention(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, is_causal=False):
        stride_qb, stride_qq, stride_qd = q.stride()
        stride_kb, stride_kk, stride_kd = k.stride()
        stride_vb, stride_vk, stride_vd = v.stride()
        
        batch_size = q.size(0)
        N_QUERIES = q.size(-2)
        N_KEYS = k.size(-2)
        d_k = q.size(-1)
        
        o = torch.zeros((batch_size, N_QUERIES, d_k), dtype=torch.float32, device="cuda")
        l = torch.zeros((batch_size, N_QUERIES), dtype=torch.float32, device="cuda")
        
        stride_ob, stride_oq, stride_od = o.stride()
        stride_lb, stride_lq = l.stride()
        
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32
        D = 16 if d_k < 16 else triton.next_power_of_2(d_k)
        
        scale = 1.0/d_k**0.5
        grid = (batch_size, triton.cdiv(N_QUERIES, Q_TILE_SIZE))
        flash_fwd_kernel[grid](
            q, k, v, o, l,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_lb, stride_lq,
            N_QUERIES, N_KEYS, scale, D, Q_TILE_SIZE, K_TILE_SIZE,
            is_causal)
        
        ctx.save_for_backward(q, k, v, o, l)
        ctx.is_causal = is_causal
        
        return o
    
    @staticmethod
    def backward(ctx, grad_output):
        pass