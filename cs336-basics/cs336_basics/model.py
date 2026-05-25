import torch
from torch import device, nn
from einops import rearrange, einsum
import torch.cuda.nvtx as nvtx

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        weight = torch.zeros((out_features, in_features), device=device, dtype=dtype)
        std = (2/(in_features+out_features))**0.5
        nn.init.trunc_normal_(weight, 0, std, -3*std, 3*std)
        self.weight = nn.Parameter(weight)
    
    def forward(self, x):
        return einsum(self.weight, x, "out_features in_features, ... in_features -> ... out_features")

class Embedding(nn.Module):
    def __init__(self, num_embedding, embedding_dim, device=None, dtype=None):
        super().__init__()
        weight = torch.zeros((num_embedding, embedding_dim), dtype=dtype, device=device)
        nn.init.trunc_normal_(weight, 0, 1, -3, 3)
        self.weight = nn.Parameter(weight)
    
    def forward(self, x):
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        weight = torch.ones((d_model,), device=device, dtype=dtype)
        self.weight = nn.Parameter(weight)
        self.eps = eps
    
    def forward(self, x:torch.Tensor):
        in_type = x.dtype
        d_model = x.size(-1)
        rms = (1/d_model*x.pow(2).sum(dim=-1, keepdim=True)+self.eps)**0.5
        res = x/rms*self.weight
        return res.to(in_type)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
    
    def forward(self, x):
        w1x = self.w1(x)
        return self.w2(w1x*torch.sigmoid(w1x)*self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        k = torch.arange(0, d_k, 2, device=device)
        seqs = torch.arange(0, max_seq_len, 1, device=device)
        angle = torch.outer(seqs, 1/theta**(k/d_k))
        self.register_buffer("sin_cache", torch.sin(angle), persistent=False)
        self.register_buffer("cos_cache", torch.cos(angle), persistent=False)
    
    def forward(self, x, token_position):
        cos = self.cos_cache[token_position]
        sin = self.sin_cache[token_position]
        pair = rearrange(x, "... (d two) -> ... d two", two=2)
        even = pair[..., 0]
        odd = pair[..., 1]
        new_even = cos*even - sin*odd
        new_odd = sin*even + cos*odd
        new_pair = torch.stack((new_even, new_odd), dim=-1)
        return rearrange(new_pair, "... d two -> ... (d two)")

def softmax(x:torch.Tensor, dim=-1):
    xmax = torch.amax(x, dim=-1, keepdim=True)
    xexp = (x-xmax).exp()
    return xexp / xexp.sum(dim, keepdim=True)

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = einsum(q, k, "... query d_k, ... key d_k -> ... query key")/d_k**0.5
    if mask is not None:
        scores = torch.where(mask, scores, float('-inf'))
    scores = softmax(scores, -1)
    return einsum(scores, v, "... query key, ... key d_k -> ... query d_k")

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(q, k, v, mask = None):
    d_k = q.size(-1)
    with nvtx.range("attention scores"):
        scores = einsum(q, k, "... query d_k, ... key d_k -> ... query key")/d_k**0.5
    with nvtx.range("mask"):
        scores = torch.where(mask, scores, float('-inf'))
    with nvtx.range("softmax"):
        scores = softmax(scores, -1)
    with nvtx.range("final matmul"):
        return einsum(scores, v, "... query key, ... key d_k -> ... query d_k")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)
        self.rope = rope
        self.num_heads = num_heads
    
    def forward(self, x:torch.Tensor, token_positions=None):
        q = rearrange(self.q_proj(x), "... context_length (num_heads d_k) -> ... num_heads context_length d_k", num_heads=self.num_heads)
        k = rearrange(self.k_proj(x), "... context_length (num_heads d_k) -> ... num_heads context_length d_k", num_heads=self.num_heads)
        v = rearrange(self.v_proj(x), "... context_length (num_heads d_k) -> ... num_heads context_length d_k", num_heads=self.num_heads)
        if token_positions is not None and self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        context_length = q.size(-2)
        mask = torch.tril(torch.ones((context_length, context_length),device=x.device, dtype=torch.bool))
        qkv = scaled_dot_product_attention(q, k, v, mask)
        res = rearrange(qkv, "... num_heads context_length d_k -> ... context_length (num_heads d_k)")
        return self.output_proj(res)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
    
    def forward(self, x, token_positions=None):
        y = self.ln1(x)
        y = self.attn(y, token_positions)
        x = x + y
        y = self.ln2(x)
        y = self.ffn(y)
        return x + y

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, d_ff, context_length, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, d_ff, num_heads, rope, device, dtype) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)
    
    def forward(self, x, token_positions=None):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)