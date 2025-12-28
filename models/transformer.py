"""
Transformer model - supports ablation studies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .modules import MultiHeadAttention, get_norm_layer, get_positional_encoding


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_rmsnorm=False, use_relative_pos=False):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout, use_relative_pos)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        norm_type = 'rmsnorm' if use_rmsnorm else 'layernorm'
        self.norm1 = get_norm_layer(norm_type, d_model)
        self.norm2 = get_norm_layer(norm_type, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_rmsnorm=False, use_relative_pos=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_relative_pos)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        norm_type = 'rmsnorm' if use_rmsnorm else 'layernorm'
        self.norm1 = get_norm_layer(norm_type, d_model)
        self.norm2 = get_norm_layer(norm_type, d_model)
        self.norm3 = get_norm_layer(norm_type, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), enc_out, enc_out, src_mask))
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, max_len=512,
                 dropout=0.1, use_rmsnorm=False, pos_encoding='absolute',
                 pad_idx=0, sos_idx=1, eos_idx=2, share_embedding=False):
        super().__init__()
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.d_model = d_model

        use_relative = pos_encoding == 'relative'
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.scale = math.sqrt(d_model)

        self.pos_enc = get_positional_encoding(pos_encoding, d_model, max_len=max_len, dropout=dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, use_rmsnorm, use_relative)
            for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, use_rmsnorm, use_relative)
            for _ in range(num_decoder_layers)])

        norm_type = 'rmsnorm' if use_rmsnorm else 'layernorm'
        self.enc_norm = get_norm_layer(norm_type, d_model)
        self.dec_norm = get_norm_layer(norm_type, d_model)
        self.output = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        if share_embedding and src_vocab_size == tgt_vocab_size:
            self.tgt_embed = self.src_embed

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_src_mask(self, src):
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def create_tgt_mask(self, tgt):
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        causal = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).unsqueeze(0).unsqueeze(0)
        return pad_mask & (causal == 1)

    def encode(self, src, src_mask=None):
        if src_mask is None:
            src_mask = self.create_src_mask(src)
        x = self.src_embed(src) * self.scale
        if self.pos_enc:
            x = self.pos_enc(x)
        else:
            x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.enc_norm(x)

    def decode(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        if tgt_mask is None:
            tgt_mask = self.create_tgt_mask(tgt)
        x = self.tgt_embed(tgt) * self.scale
        if self.pos_enc:
            x = self.pos_enc(x)
        else:
            x = self.dropout(x)
        for layer in self.decoder_layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.output(self.dec_norm(x))

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if src_mask is None:
            src_mask = self.create_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.create_tgt_mask(tgt)
        enc_out = self.encode(src, src_mask)
        return self.decode(tgt, enc_out, tgt_mask, src_mask)


def build_transformer_model(config):
    return Transformer(
        src_vocab_size=config.get('src_vocab_size', 5000),
        tgt_vocab_size=config.get('tgt_vocab_size', 5000),
        d_model=config.get('d_model', 512),
        num_heads=config.get('num_heads', 8),
        num_encoder_layers=config.get('num_encoder_layers', config.get('num_layers', 6)),
        num_decoder_layers=config.get('num_decoder_layers', config.get('num_layers', 6)),
        d_ff=config.get('d_ff', 2048),
        max_len=config.get('max_len', 512),
        dropout=config.get('dropout', 0.1),
        use_rmsnorm=config.get('use_rmsnorm', False),
        pos_encoding=config.get('pos_encoding', 'absolute'),
        pad_idx=config.get('pad_idx', 0),
        sos_idx=config.get('sos_idx', 1),
        eos_idx=config.get('eos_idx', 2),
        share_embedding=config.get('share_embedding', False)
    )
