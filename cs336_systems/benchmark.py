import argparse
import torch
from cs336_basics import model, nn_utils, optimizer

def gen_batch(batch_size, context_length, vocab_size, device: str) -> (torch.Tensor, torch.Tensor):
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size,), device=device)
    return (x, y)    
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="small", choices=["small", "medium", "large", "xl", "10B"],)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    
    context_length = 512
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
            "d_ff": 1228,
            "num_layers": 50,
            "num_heads": 36
        }
    }

    args = parser.parse_args()
    config = model_config[args.size]
    
    print(f'{args=} {config=}')
    
    m = model.BasicsTransformerLM(
        vocab_size, 
        context_length,
        config["d_model"],
        config["num_layers"],
        config["num_heads"],
        config["d_ff"],
    )
    opt = optimizer.AdamW(m.parameters())
    
    x, y = gen_batch(batch_size, context_length, vocab_size, device)
    print(f'{x=} {y=}')
    for _ in range(args.warmup + args.steps):
        inputs = m(x)
        loss = nn_utils.cross_entropy(inputs, y)
        print(f'{x=} {inputs=} {loss=}')
        opt.zero_grad()
        loss.backward()
        opt.step()

main()