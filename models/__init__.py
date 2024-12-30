from .llama_compile import LLaMA as LLaMACompile
from .llama import LLaMA
from .transformer import Transformer
from .bert import Bert
from .gpt_neox import GPTNeoX
from .qwen2 import Qwen2

model_map = {
    'LLaMA': LLaMA,
    'LLaMACompile': LLaMACompile,
    'Transformer': Transformer,
    'Bert': Bert,
    'GPTNeoX': GPTNeoX,
    'Qwen2': Qwen2
}

__all__ = [
    'LLaMACompile',
    'LLaMA',
    'Transformer',
    'Bert',
    'GPTNeoX',
    'Qwen2',
    'model_map'
]
