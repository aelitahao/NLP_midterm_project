"""
Neural network modules - Attention, Normalization, Positional Encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== Attention Mechanisms ====================

class DotAttention(nn.Module):
    """Dot-Product Attention"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.scale = math.sqrt(hidden_size)

    def forward(self, query, keys, mask=None):
        if query.dim() == 2:
            query = query.unsqueeze(1)
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        context = torch.bmm(weights, keys)
        return context.squeeze(1), weights.squeeze(1)


class GeneralAttention(nn.Module):
    """Luong General Attention"""
    def __init__(self, query_size: int, key_size: int):
        super().__init__()
        self.W = nn.Linear(key_size, query_size, bias=False)

    def forward(self, query, keys, mask=None):
        if query.dim() == 2:
            query = query.unsqueeze(1)
        scores = torch.bmm(query, self.W(keys).transpose(1, 2))
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        context = torch.bmm(weights, keys)
        return context.squeeze(1), weights.squeeze(1)


class AdditiveAttention(nn.Module):
    """Bahdanau Additive Attention"""
    def __init__(self, query_size: int, key_size: int, hidden_size: int):
        super().__init__()
        self.W1 = nn.Linear(query_size, hidden_size, bias=False)
        self.W2 = nn.Linear(key_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, keys, mask=None):
        seq_len = keys.size(1)
        q = self.W1(query).unsqueeze(1).expand(-1, seq_len, -1)
        k = self.W2(keys)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 use_relative_pos: bool = False, max_relative_pos: int = 128):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_relative_pos = use_relative_pos

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if use_relative_pos:
            self.max_relative_pos = max_relative_pos
            self.relative_pos_embedding = nn.Embedding(2 * max_relative_pos + 1, num_heads)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if self.use_relative_pos:
            seq_len = query.size(1)
            pos = torch.arange(seq_len, device=query.device)
            rel_pos = torch.clamp(pos.unsqueeze(0) - pos.unsqueeze(1),
                                  -self.max_relative_pos, self.max_relative_pos)
            rel_pos = rel_pos + self.max_relative_pos
            bias = self.relative_pos_embedding(rel_pos).permute(2, 0, 1)
            scores = scores + bias.unsqueeze(0)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)
        context = torch.matmul(weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)


def get_attention(attention_type: str, **kwargs):
    """Attention factory function"""
    if attention_type == 'dot':
        return DotAttention(kwargs.get('hidden_size', 256))
    elif attention_type == 'general':
        return GeneralAttention(kwargs.get('query_size', 256), kwargs.get('key_size', 256))
    elif attention_type in ['additive', 'concat']:
        return AdditiveAttention(kwargs.get('query_size', 256),
                                  kwargs.get('key_size', 256),
                                  kwargs.get('attention_hidden_size', 256))
    raise ValueError(f"Unknown attention type: {attention_type}")


# ==================== Normalization ====================

class RMSNorm(nn.Module):
    """RMS Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.gamma * x / rms


def get_norm_layer(norm_type: str, dim: int):
    """Normalization layer factory"""
    if norm_type == 'rmsnorm':
        return RMSNorm(dim)
    return nn.LayerNorm(dim)


# ==================== Positional Encoding ====================

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def get_positional_encoding(pos_type: str, d_model: int, **kwargs):
    """Positional encoding factory"""
    if pos_type in ['absolute', 'sinusoidal']:
        return SinusoidalPositionalEncoding(d_model, kwargs.get('max_len', 5000), kwargs.get('dropout', 0.1))
    return None
