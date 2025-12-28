"""
RNN Seq2Seq model - GRU/LSTM + Attention
"""

import torch
import torch.nn as nn
from .modules import get_attention


class Encoder(nn.Module):
    """RNN Encoder"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2,
                 rnn_type='gru', dropout=0.1, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        rnn_cls = nn.GRU if rnn_type == 'gru' else nn.LSTM
        self.rnn = rnn_cls(embed_size, hidden_size, num_layers,
                           dropout=dropout if num_layers > 1 else 0,
                           bidirectional=bidirectional, batch_first=True)

        if bidirectional:
            self.projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src, src_lengths=None):
        embedded = self.dropout(self.embedding(src))

        if src_lengths is not None:
            src_lengths = src_lengths.clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
            outputs, hidden = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, hidden = self.rnn(embedded)

        if isinstance(hidden, tuple):
            hidden = hidden[0]

        if self.bidirectional:
            batch_size = hidden.size(1)
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
            hidden = hidden.transpose(1, 2).contiguous().view(self.num_layers, batch_size, -1)
            hidden = torch.tanh(self.projection(hidden))

        return outputs, hidden


class AttentionDecoder(nn.Module):
    """RNN Decoder with Attention"""
    def __init__(self, vocab_size, embed_size, hidden_size, encoder_hidden_size,
                 num_layers=2, rnn_type='gru', attention_type='general', dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        rnn_cls = nn.GRU if rnn_type == 'gru' else nn.LSTM
        self.rnn = rnn_cls(embed_size + encoder_hidden_size, hidden_size,
                           num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)

        self.attention = get_attention(attention_type, hidden_size=hidden_size,
                                        query_size=hidden_size, key_size=encoder_hidden_size,
                                        attention_hidden_size=hidden_size)
        self.output = nn.Linear(hidden_size + encoder_hidden_size, vocab_size)

    def forward(self, token, hidden, encoder_outputs, mask=None, context=None):
        embedded = self.dropout(self.embedding(token))

        if context is None:
            context = torch.zeros(token.size(0), encoder_outputs.size(2), device=token.device)

        rnn_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)

        context, weights = self.attention(output, encoder_outputs, mask)
        output = self.output(torch.cat([output, context], dim=1))

        return output, hidden, context, weights


class RNNSeq2Seq(nn.Module):
    """RNN Seq2Seq Model"""
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=256, hidden_size=512,
                 num_layers=2, rnn_type='gru', attention_type='general', dropout=0.1,
                 bidirectional=False, pad_idx=0, sos_idx=1, eos_idx=2):
        super().__init__()
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.tgt_vocab_size = tgt_vocab_size

        enc_hidden = hidden_size * 2 if bidirectional else hidden_size
        self.encoder = Encoder(src_vocab_size, embed_size, hidden_size,
                               num_layers, rnn_type, dropout, bidirectional)
        self.decoder = AttentionDecoder(tgt_vocab_size, embed_size, hidden_size,
                                         enc_hidden, num_layers, rnn_type, attention_type, dropout)

    def create_mask(self, src):
        # True indicates padding positions (to be masked)
        return src == self.pad_idx

    def forward(self, src, tgt, src_lengths=None, teacher_forcing_ratio=1.0):
        batch_size, tgt_len = tgt.shape
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        mask = self.create_mask(src)

        outputs = torch.zeros(batch_size, tgt_len - 1, self.tgt_vocab_size, device=src.device)
        input_token = tgt[:, 0]
        context = None

        for t in range(1, tgt_len):
            output, hidden, context, _ = self.decoder(input_token, hidden, encoder_outputs, mask, context)
            outputs[:, t - 1] = output

            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = tgt[:, t]
            else:
                input_token = output.argmax(dim=1)

        return outputs

    def encode(self, src, src_lengths=None):
        return self.encoder(src, src_lengths)

    def decode_step(self, token, hidden, encoder_outputs, mask, context=None):
        return self.decoder(token, hidden, encoder_outputs, mask, context)


def build_rnn_model(config):
    return RNNSeq2Seq(
        src_vocab_size=config.get('src_vocab_size', 5000),
        tgt_vocab_size=config.get('tgt_vocab_size', 5000),
        embed_size=config.get('embed_size', 256),
        hidden_size=config.get('hidden_size', 512),
        num_layers=config.get('num_layers', 2),
        rnn_type=config.get('rnn_type', 'gru'),
        attention_type=config.get('attention_type', 'general'),
        dropout=config.get('dropout', 0.1),
        bidirectional=config.get('bidirectional', False),
        pad_idx=config.get('pad_idx', 0),
        sos_idx=config.get('sos_idx', 1),
        eos_idx=config.get('eos_idx', 2)
    )
