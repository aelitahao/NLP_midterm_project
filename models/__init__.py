"""NMT model collection"""

from .rnn import RNNSeq2Seq, build_rnn_model
from .transformer import Transformer, build_transformer_model
from .decoder import greedy_decode, beam_search

try:
    from .t5_wrapper import T5Wrapper, build_t5_model
    HAS_T5 = True
except ImportError:
    HAS_T5 = False
    T5Wrapper = None
    build_t5_model = None


def build_model(model_type: str, config: dict):
    """Model factory"""
    if model_type == 'rnn':
        return build_rnn_model(config)
    elif model_type == 'transformer':
        return build_transformer_model(config)
    elif model_type == 't5':
        if not HAS_T5:
            raise ImportError("T5 requires transformers library")
        return build_t5_model(config)
    raise ValueError(f"Unknown model: {model_type}")
