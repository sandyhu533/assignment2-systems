import torch
from cs336_basics import model
from autograd_inspect import *

device = "cpu"


batch_size = 8
num_heads = 1
context_length = 1024
d_k = 32

q = torch.rand((batch_size, num_heads, context_length, d_k),device=device, requires_grad=True)
k = torch.rand((batch_size, num_heads, context_length, d_k),device=device, requires_grad=True)
v = torch.rand((batch_size, num_heads, context_length, d_k),device=device,requires_grad=True)
mask = torch.tril(torch.ones((context_length, context_length), dtype=torch.bool,device=device))
att = model.scaled_dot_product_attention(
    q, k, v, mask
)
v = att.sum() # N/A
print_graph(v)
print()
print_saved_activations(v)
print()
trace_backward(v, show_values=True)
v.backward()
