from collections import defaultdict

def round_d_ff(d_model: int) -> int:
    return (8/3*d_model//64)*64

def transformer_params(vocab_size, num_layers, d_model, num_heads, d_ff=None) -> dict:
    if d_ff == None: d_ff = round_d_ff(d_model)
    dic = defaultdict(int)
    dic['token_embedding'] = vocab_size*d_model
    dic['attention'] = d_model*d_model*4*num_layers
    dic['ffn'] = d_model*d_ff*3*num_layers
    dic['norms'] = d_model*(2*num_layers+1)
    dic['lm_head'] = d_model*vocab_size
    dic['total'] = sum(dic.values())
    return dic

def transformer_forward_flops(batch_size, context_length, vocab_size,
                              num_layers, d_model, num_heads, d_ff=None) -> dict:
    dic = defaultdict(int)
    dic['attention_qkvo'] = 4*2*batch_size*context_length*d_model*d_model*num_layers
    dic['attention_scores'] = 2*batch_size*context_length*context_length*d_model*num_layers
    dic['attention_softmax_v'] = 2*batch_size*context_length*context_length*d_model*num_layers
    dic['ffn'] = 3*2*batch_size*context_length*d_model*d_ff*num_layers
    dic['lm_head'] = 2*batch_size*context_length*vocab_size*d_model
    dic['total'] = sum(dic.values())
    return dic

def activation_elements(batch_size, context_length, vocab_size,
                        num_layers, d_model, num_heads, d_ff=None) -> dict:
    dic = defaultdict(int)
    dic['attention'] = 5*batch_size*context_length*d_model*num_layers #qkvo + qkv result
    dic['attention_T2'] = 2*batch_size*context_length*context_length*num_heads*num_layers # scores, softmax
    dic['ffn'] = 3*batch_size*context_length*d_ff + batch_size*context_length*d_model # w1,w2,w3,silu
    dic['ffn'] *= num_layers
    dic['norms'] = batch_size*context_length*d_model*(num_layers*2+1)
    dic['logits'] = batch_size*context_length*vocab_size
    dic['total'] = sum(dic.values())
    return dic

def adamw_peak_memory_bytes(batch_size, context_length, vocab_size,
                            num_layers, d_model, num_heads,
                            d_ff=None, bytes_per_element=4) -> dict:
    dic = defaultdict(int)
    dic['params'] = transformer_params(vocab_size, num_layers, d_model, num_heads, d_ff)['total']*4
    dic['gradients'] = dic['params']
    dic['optimizer'] = 2*dic['params']
    dic['activations'] = activation_elements(batch_size, context_length, vocab_size, num_layers, d_model, num_heads, d_ff)['total']*4
    dic['total'] = sum(dic.values())
    return dic

def adamw_step_flops(vocab_size, num_layers, d_model, num_heads, d_ff=None) -> dict:
    params = transformer_params(vocab_size, num_layers, d_model, num_heads, d_ff)['total']
    dic = defaultdict(int)
    dic['weight_decay'] = 2*params
    dic['m_update'] = 3*params
    dic['v_update'] = 4*params
    dic['param_update'] = 5*params
    dic['total'] = sum(dic.values())
    return dic


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
 
context_length = 1024
vocab_size=10000
batch_size=4
    
def human_bytes(n: int | float, binary: bool = False, precision: int = 2) -> str:
    """
    >>> human_bytes(1536)                # '1.50 KB'
    >>> human_bytes(1536, binary=True)   # '1.50 KiB'
    >>> human_bytes(1_500_000_000)       # '1.50 GB'
    >>> human_bytes(0)                   # '0 B'
    """
    base = 1024 if binary else 1000
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"] if binary else \
            ["B", "KB",  "MB",  "GB",  "TB",  "PB"]
    if n == 0:
        return "0 B"
    sign = "-" if n < 0 else ""
    n = abs(n)
    i = 0
    while n >= base and i < len(units) - 1:
        n /= base
        i += 1
    return f"{sign}{n:.{precision}f} {units[i]}"

for model, config in model_config.items():
    print(f'{model=} {config=}')
    d = adamw_peak_memory_bytes(
        batch_size, context_length, vocab_size, config["num_layers"],
        config["d_model"], config["num_heads"], config["d_ff"]
        )
    readable = {k: human_bytes(v, binary=True) for k, v in d.items()}
    print(readable)
