import torch
import torch.nn as nn
import math
from typing import Optional
from torch import Tensor
from .utils import Config


class GPTNeoX(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_layers))

        self.norm = nn.LayerNorm(config.embedding_dim, eps=config.norm_eps)
        self.linear = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        self.max_seq_len = -1
        self.causal_mask = None
        self.freq_cis = None
        self.use_cache = False

    def setup_caches(self, max_seq_len: int, use_cache=False):
        self.max_seq_len = max_seq_len
        dtype = self.linear.weight.dtype

        if hasattr(self.linear, 'scales'):
            dtype = self.linear.scales.dtype
        elif hasattr(self.linear, 'scales_and_zeros'):
            dtype = self.linear.scales_and_zeros.dtype

        self.use_cache = use_cache
        for layer in self.layers:
            if self.use_cache:
                layer.self_attn.kv_cache = KVCache()
            else:
                layer.self_attn.kv_cache = None

        rotary_dim = int((self.config.embedding_dim // self.config.num_heads) * self.config.rotary_pct)
        self.freq_cis = precompute_freqs_cis(self.max_seq_len,
                                             rotary_dim,
                                             self.config.rope_base,
                                             dtype,
                                             self.config.rope_scaling)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool))

    def forward(self, input_ids: Tensor) -> Tensor:
        seq_len = input_ids.size(1)
        pre_len = 0
        if self.use_cache and self.layers[0].self_attn.kv_cache.k_cache is not None:
            pre_len = self.layers[0].self_attn.kv_cache.k_cache.size(2)
            input_ids = input_ids[:, pre_len:]

        attention_mask = self.causal_mask[pre_len: seq_len, : seq_len]
        freqs_cis = self.freq_cis[pre_len: seq_len]

        x = self.word_embeddings(input_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attention_mask, freqs_cis)
        x = self.norm(x)
        logits = self.linear(x)

        return logits


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiheadAttention(config)
        self.ff = FeedForward(config)
        self.attn_norm = nn.LayerNorm(config.embedding_dim, eps=config.norm_eps)
        self.ff_norm = nn.LayerNorm(config.embedding_dim, eps=config.norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout_rate)
        self.ff_dropout = nn.Dropout(config.dropout_rate)

        self.use_parallel_residual = config.use_parallel_residual

    def forward(self, x: Tensor, attention_mask: Tensor, freq_cis: Tensor):
        attn_output = self.self_attn(self.attn_norm(x), attention_mask, freq_cis)
        attn_output = self.attn_dropout(attn_output)
        if self.use_parallel_residual:
            ff_output = self.ff(self.ff_norm(x))
            ff_output = self.ff_dropout(ff_output)
            output = x + ff_output + attn_output
        else:
            attn_output = x + attn_output
            ff_output = self.ff(self.ff_norm(attn_output))
            ff_output = self.ff_dropout(ff_output)
            output = ff_output + attn_output

        return output


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.embedding_dim, config.feedforward_dim)
        self.linear2 = nn.Linear(config.feedforward_dim, config.embedding_dim)
        self.act = nn.GELU()

    def forward(self, x: Tensor):
        return self.linear2(self.act(self.linear1(x)))


class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = self.embedding_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embedding_dim, "embedding_dim must be divisible by num_heads"
        self.rotary_dim = int(self.head_dim * config.rotary_pct)

        self.linear_qkv = nn.Linear(self.embedding_dim, 3 * self.embedding_dim)

        self.linear = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.kv_cache = None

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + 'linear_q.weight' in state_dict:
            wq = state_dict.pop(prefix + 'linear_q.weight')
            wk = state_dict.pop(prefix + 'linear_k.weight')
            wv = state_dict.pop(prefix + 'linear_v.weight')
            bq = state_dict.pop(prefix + 'linear_q.bias')
            bk = state_dict.pop(prefix + 'linear_k.bias')
            bv = state_dict.pop(prefix + 'linear_v.bias')
            state_dict[prefix + 'linear_qkv.weight'] = torch.cat([wq, wk, wv])
            state_dict[prefix + 'linear_qkv.bias'] = torch.cat([bq, bk, bv])
        if prefix + 'linear_q.scales' in state_dict:
            scale_q = state_dict.pop(prefix + 'linear_q.scales')
            scale_k = state_dict.pop(prefix + 'linear_k.scales')
            scale_v = state_dict.pop(prefix + 'linear_v.scales')
            state_dict[prefix + 'linear_qkv.scales'] = torch.cat([scale_q, scale_k, scale_v])

    def forward(self, x: Tensor, attention_mask: Tensor, freqs_cis: Tensor):
        bs, seq_len = x.shape[:2]

        query, key, value = self.linear_qkv(x).split([self.embedding_dim, self.embedding_dim, self.embedding_dim], dim=-1)

        query = query.view(bs, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(bs, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(bs, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        query_rot, query_n_rot = query.split([self.rotary_dim, self.head_dim - self.rotary_dim], dim=-1)
        key_rot, key_n_rot = key.split([self.rotary_dim, self.head_dim - self.rotary_dim], dim=-1)

        # query_rot -> bs, num_heads, seq_len, rotary_dim
        # freqs_cis -> seq_len, head_dim / 2, 2
        query_rot = apply_rotate_emb(query_rot, freqs_cis)
        key_rot = apply_rotate_emb(key_rot, freqs_cis)

        query = torch.cat([query_rot, query_n_rot], dim=-1)
        key = torch.cat([key_rot, key_n_rot], dim=-1)

        if self.kv_cache:
            key, value = self.kv_cache.update(key, value)

        attention_score = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        attention_score.masked_fill_(~attention_mask, -torch.inf)

        attention_score = torch.softmax(attention_score, dim=-1)
        output = attention_score @ value
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, self.embedding_dim)

        output = self.linear(output)

        return output


class KVCache(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_cache = None  # bs, num_heads, ~, head_dim
        self.v_cache = None

    def update(self, k_val: Tensor, v_val: Tensor):
        # val -> bs, num_heads, seq_len, head_dim
        if self.k_cache is None:
            self.k_cache = k_val
            self.v_cache = v_val
        else:
            self.k_cache = torch.cat([self.k_cache, k_val], dim=2)
            self.v_cache = torch.cat([self.v_cache, v_val], dim=2)

        return self.k_cache, self.v_cache


def apply_rope_scaling(freqs: Tensor, rope_scaling: Optional[dict] = None):
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(max_seq_len: int, dim: int, base: int = 10000, dtype: torch.dtype = torch.bfloat16,
                         rope_scaling: Optional[dict] = None) -> Tensor:
    freqs = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    position = torch.arange(0, max_seq_len, dtype=torch.float)
    # position -> max_seq_len, 1  freqs -> dim / 2
    freqs = torch.outer(position, freqs)  # max_seq_len, dim  / 2
    freqs_cis = torch.stack([freqs.cos(), freqs.sin()], dim=-1)
    return freqs_cis.to(dtype=dtype)  # max_seq_len, dim // 2, 2


def apply_rotate_emb(x: Tensor, freqs_cis):
    # x -> bs, num_heads, seq_len, dim
    # freqs_cis -> seq_len, dim // 2, 2
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    cos = freqs_cis[..., 0]  # seq_len, dim // 2
    sin = freqs_cis[..., 1]  # seq_len, dim // 2

    x_out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

    return x_out.type_as(x)
