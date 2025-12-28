"""Utility module - evaluation metrics and visualization"""

import math
import re
from collections import Counter
from typing import List, Dict
import matplotlib.pyplot as plt
import os


# ==================== BLEU Evaluation ====================

def tokenize(text: str, lang: str = 'en') -> List[str]:
    if lang == 'zh':
        return list(text.replace(' ', ''))
    text = re.sub(r'([.,!?;:])', r' \1 ', text.lower())
    return text.split()


def corpus_bleu(refs_list, hyps, max_n=4) -> Dict[str, float]:
    """Corpus-level BLEU"""
    total_hyp_len = total_ref_len = 0
    clipped = [0] * max_n
    total = [0] * max_n

    for refs, hyp in zip(refs_list, hyps):
        hyp_len = len(hyp)
        total_hyp_len += hyp_len
        ref_lens = [len(r) for r in refs]
        total_ref_len += min(ref_lens, key=lambda x: (abs(x - hyp_len), x))

        for n in range(1, max_n + 1):
            hyp_ngrams = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp) - n + 1))
            if not hyp_ngrams:
                continue
            max_ref = Counter()
            for ref in refs:
                ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref) - n + 1))
                for ng in hyp_ngrams:
                    max_ref[ng] = max(max_ref[ng], ref_ngrams[ng])
            for ng, c in hyp_ngrams.items():
                clipped[n-1] += min(c, max_ref[ng])
                total[n-1] += c

    precs = [clipped[n] / total[n] if total[n] else 0 for n in range(max_n)]
    if min(precs) > 0:
        score = math.exp(sum(math.log(p) for p in precs) / max_n)
    else:
        score = 0

    bp = 1.0 if total_hyp_len >= total_ref_len else (
        math.exp(1 - total_ref_len / total_hyp_len) if total_hyp_len else 0)

    return {'bleu': bp * score * 100, 'bleu-1': precs[0]*100, 'bleu-2': precs[1]*100,
            'bleu-3': precs[2]*100, 'bleu-4': precs[3]*100, 'bp': bp}


class BLEUScorer:
    def __init__(self, lang='en'):
        self.lang = lang
        self.refs = []
        self.hyps = []

    def add(self, ref, hyp):
        self.refs.append([tokenize(ref, self.lang)])
        self.hyps.append(tokenize(hyp, self.lang))

    def add_batch(self, refs, hyps):
        for r, h in zip(refs, hyps):
            self.add(r, h)

    def compute(self):
        return corpus_bleu(self.refs, self.hyps)


# ==================== Training Curve Visualization ====================

def parse_training_log(log_file: str) -> Dict[str, List[float]]:
    """
    Parse training and validation loss from training log file

    Args:
        log_file: Training log file path

    Returns:
        Dictionary containing epochs, train_losses, val_losses
    """
    epochs = []
    train_losses = []
    val_losses = []

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Match lines like "Epoch 0 | Time: 49.3s | Train Loss: 5.7366 | Val Loss: 5.1109"
            # or "Epoch 0 | Time: 138.1s | Train: 3.2974 | Val: 3.7356" (T5 format)
            if ('| Train Loss:' in line and '| Val Loss:' in line) or \
               ('| Train:' in line and '| Val:' in line):
                parts = line.strip().split('|')
                try:
                    # Extract epoch
                    epoch_part = parts[0].strip()
                    epoch = int(epoch_part.split()[1])

                    # Extract train loss
                    train_loss_part = parts[2].strip()
                    train_loss = float(train_loss_part.split(':')[1].strip())

                    # Extract val loss
                    val_loss_part = parts[3].strip()
                    val_loss = float(val_loss_part.split(':')[1].strip())

                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                except (IndexError, ValueError):
                    continue

    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses
    }


def plot_training_curve(log_file: str, output_path: str, title: str = "Training Curve"):
    """
    Plot training curve (Train Loss vs Valid Loss)

    Args:
        log_file: Training log file path
        output_path: Output image path
        title: Chart title
    """
    # Parse log
    data = parse_training_log(log_file)

    if not data['epochs']:
        print(f"Warning: No training data found in {log_file}")
        return

    # Create chart
    plt.figure(figsize=(10, 6))
    plt.plot(data['epochs'], data['train_losses'], label='Train Loss', marker='o', linewidth=2, markersize=4)
    plt.plot(data['epochs'], data['val_losses'], label='Valid Loss', marker='s', linewidth=2, markersize=4)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Mark best validation loss
    min_val_idx = data['val_losses'].index(min(data['val_losses']))
    min_val_loss = data['val_losses'][min_val_idx]
    min_val_epoch = data['epochs'][min_val_idx]
    plt.annotate(f'Best Val Loss: {min_val_loss:.4f}\n(Epoch {min_val_epoch})',
                xy=(min_val_epoch, min_val_loss),
                xytext=(min_val_epoch + len(data['epochs']) * 0.1, min_val_loss + (max(data['val_losses']) - min(data['val_losses'])) * 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    plt.tight_layout()

    # Save chart
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training curve saved to: {output_path}")
    plt.close()


def generate_all_curves(log_dir: str = 'log', output_dir: str = 'figures'):
    """
    Batch generate training curves for all models

    Args:
        log_dir: Log file directory
        output_dir: Output image directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # RNN model configs
    rnn_configs = [
        ('rnn_general_10k_train.log', 'RNN General Attention (10k Dataset)', 'rnn_general_10k.png'),
        ('rnn_general_100k_train.log', 'RNN General Attention (100k Dataset)', 'rnn_general_100k.png'),
        ('rnn_dot_10k_train.log', 'RNN Dot Attention (10k Dataset)', 'rnn_dot_10k.png'),
        ('rnn_dot_100k_train.log', 'RNN Dot Attention (100k Dataset)', 'rnn_dot_100k.png'),
        ('rnn_additive_10k_train.log', 'RNN Additive Attention (10k Dataset)', 'rnn_additive_10k.png'),
        ('rnn_additive_100k_train.log', 'RNN Additive Attention (100k Dataset)', 'rnn_additive_100k.png'),
    ]

    # Transformer model configs
    transformer_configs = [
        ('transformer_absolute_10k_train.log', 'Transformer Absolute PE (10k Dataset)', 'transformer_absolute_10k.png'),
        ('transformer_absolute_100k_train.log', 'Transformer Absolute PE (100k Dataset)', 'transformer_absolute_100k.png'),
        ('transformer_relative_10k_train.log', 'Transformer Relative PE (10k Dataset)', 'transformer_relative_10k.png'),
        ('transformer_relative_100k_train.log', 'Transformer Relative PE (100k Dataset)', 'transformer_relative_100k.png'),
        ('transformer_rmsnorm_10k_train.log', 'Transformer RMSNorm (10k Dataset)', 'transformer_rmsnorm_10k.png'),
        ('transformer_rmsnorm_100k_train.log', 'Transformer RMSNorm (100k Dataset)', 'transformer_rmsnorm_100k.png'),
    ]

    # T5 model configs
    t5_configs = [
        ('t5_base_100k_train.log', 'T5-Base (100k Dataset)', 't5_base_100k.png'),
        ('t5_large_100k_train.log', 'T5-Large (100k Dataset)', 't5_large_100k.png'),
    ]

    # Generate RNN curves
    print("\n=== Generating RNN Training Curves ===")
    for log_file, title, output_file in rnn_configs:
        log_path = os.path.join(log_dir, log_file)
        output_path = os.path.join(output_dir, output_file)
        if os.path.exists(log_path):
            plot_training_curve(log_path, output_path, title)
        else:
            print(f"Warning: {log_path} not found, skipping...")

    # Generate Transformer curves
    print("\n=== Generating Transformer Training Curves ===")
    for log_file, title, output_file in transformer_configs:
        log_path = os.path.join(log_dir, log_file)
        output_path = os.path.join(output_dir, output_file)
        if os.path.exists(log_path):
            plot_training_curve(log_path, output_path, title)
        else:
            print(f"Warning: {log_path} not found, skipping...")

    # Generate T5 curves
    print("\n=== Generating T5 Training Curves ===")
    for log_file, title, output_file in t5_configs:
        log_path = os.path.join(log_dir, log_file)
        output_path = os.path.join(output_dir, output_file)
        if os.path.exists(log_path):
            plot_training_curve(log_path, output_path, title)
        else:
            print(f"Warning: {log_path} not found, skipping...")

    print(f"\n=== All training curves generated in '{output_dir}/' ===")


# ==================== Attention Weight Visualization ====================

import torch
import numpy as np


def plot_attention_heatmap(attention_weights, src_tokens, tgt_tokens, output_path, title="Attention Weights"):
    """
    Plot attention weight heatmap

    Args:
        attention_weights: Attention weight matrix [tgt_len, src_len]
        src_tokens: Source sequence token list
        tgt_tokens: Target sequence token list
        output_path: Output image path
        title: Chart title
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()

    # Set Chinese font support
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw heatmap
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')

    # Set axes
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(tgt_tokens, fontsize=10)

    ax.set_xlabel('Source Tokens', fontsize=12)
    ax.set_ylabel('Target Tokens', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add color bar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=11)

    plt.tight_layout()

    # Save chart
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Attention heatmap saved to: {output_path}")
    plt.close()


@torch.no_grad()
def get_rnn_attention_weights(model, src, src_tokens, vocab_en, max_len=50, sos_idx=1, eos_idx=2):
    """
    Get RNN model attention weights

    Args:
        model: RNN Seq2Seq model
        src: Source sequence tensor [1, src_len]
        src_tokens: Source sequence token list
        vocab_en: English vocabulary
        max_len: Maximum decoding length
        sos_idx: SOS token ID
        eos_idx: EOS token ID

    Returns:
        attention_weights: Attention weights [tgt_len, src_len]
        tgt_tokens: Generated target token list
    """
    model.eval()
    device = src.device

    # Encode
    enc_out, hidden = model.encode(src, None)
    mask = model.create_mask(src)

    # Decode and collect attention weights
    all_weights = []
    tgt_tokens = []
    token = torch.tensor([sos_idx], device=device)
    context = None

    for _ in range(max_len):
        output, hidden, context, weights = model.decode_step(token, hidden, enc_out, mask, context)
        all_weights.append(weights.squeeze(0))

        next_token = output.argmax(dim=-1)
        token_id = next_token.item()

        if token_id == eos_idx:
            break

        # Decode token
        if vocab_en and token_id < len(vocab_en.idx2word):
            tgt_tokens.append(vocab_en.idx2word[token_id])
        else:
            tgt_tokens.append(f'<{token_id}>')

        token = next_token

    if all_weights:
        attention_weights = torch.stack(all_weights, dim=0)
        # Crop to actual source sequence length
        src_len = len(src_tokens)
        attention_weights = attention_weights[:, :src_len]
        return attention_weights, tgt_tokens

    return None, []


@torch.no_grad()
def get_transformer_attention_weights(model, src, layer_idx=0, head_idx=0):
    """
    Get Transformer model self-attention weights (using hooks)

    Args:
        model: Transformer model
        src: Source sequence tensor [1, src_len]
        layer_idx: Encoder layer index to visualize
        head_idx: Attention head index to visualize

    Returns:
        attention_weights: Attention weights [src_len, src_len]
    """
    model.eval()
    attention_weights = []

    def hook_fn(module, input, output):
        # Get Q, K to compute attention scores
        query, key, value = input[0], input[1], input[2]
        batch_size = query.size(0)
        d_k = model.d_model // model.encoder_layers[0].attn.num_heads
        num_heads = model.encoder_layers[0].attn.num_heads

        # Compute Q, K
        Q = module.W_q(query).view(batch_size, -1, num_heads, d_k).transpose(1, 2)
        K = module.W_k(key).view(batch_size, -1, num_heads, d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)

        # Save weights for specified head
        attention_weights.append(weights[0, head_idx].cpu())

    # Register hook
    target_layer = model.encoder_layers[layer_idx].attn
    handle = target_layer.register_forward_hook(hook_fn)

    # Forward pass
    src_mask = model.create_src_mask(src)
    _ = model.encode(src, src_mask)

    # Remove hook
    handle.remove()

    if attention_weights:
        return attention_weights[0]
    return None


def visualize_rnn_attention(model, text, vocab_zh, vocab_en, output_path,
                            title="RNN Attention Weights", device='cuda'):
    """
    Visualize RNN model attention weights

    Args:
        model: RNN Seq2Seq model
        text: Input Chinese text
        vocab_zh: Chinese vocabulary
        vocab_en: English vocabulary
        output_path: Output image path
        title: Chart title
        device: Device
    """
    # Tokenize
    import jieba
    src_tokens = list(jieba.cut(text))

    # Encode
    src_ids = vocab_zh.encode(src_tokens)
    src = torch.tensor([src_ids], device=device)

    # Get attention weights
    attention_weights, tgt_tokens = get_rnn_attention_weights(
        model, src, src_tokens, vocab_en,
        sos_idx=model.sos_idx, eos_idx=model.eos_idx
    )

    if attention_weights is not None and len(tgt_tokens) > 0:
        plot_attention_heatmap(attention_weights, src_tokens, tgt_tokens, output_path, title)
    else:
        print(f"Warning: Could not generate attention weights for: {text}")


def visualize_transformer_attention(model, text, vocab_zh, output_path,
                                    title="Transformer Self-Attention", layer_idx=0, head_idx=0, device='cuda'):
    """
    Visualize Transformer model self-attention weights

    Args:
        model: Transformer model
        text: Input Chinese text
        vocab_zh: Chinese vocabulary
        output_path: Output image path
        title: Chart title
        layer_idx: Encoder layer index
        head_idx: Attention head index
        device: Device
    """
    # Tokenize
    import jieba
    src_tokens = list(jieba.cut(text))

    # Encode
    src_ids = vocab_zh.encode(src_tokens)
    src = torch.tensor([src_ids], device=device)

    # Get attention weights
    attention_weights = get_transformer_attention_weights(model, src, layer_idx, head_idx)

    if attention_weights is not None:
        # Self-attention: source and target are the same
        weights = attention_weights[:len(src_tokens), :len(src_tokens)]
        plot_attention_heatmap(weights, src_tokens, src_tokens, output_path,
                              f"{title} (Layer {layer_idx}, Head {head_idx})")
    else:
        print(f"Warning: Could not generate attention weights for: {text}")


def visualize_transformer_multihead(model, text, vocab_zh, output_dir,
                                    title_prefix="Transformer", layer_idx=0, device='cuda'):
    """
    Visualize Transformer multi-head self-attention weights

    Args:
        model: Transformer model
        text: Input Chinese text
        vocab_zh: Chinese vocabulary
        output_dir: Output directory
        title_prefix: Title prefix
        layer_idx: Encoder layer index
        device: Device
    """
    os.makedirs(output_dir, exist_ok=True)
    num_heads = model.encoder_layers[0].attn.num_heads

    # Tokenize
    import jieba
    src_tokens = list(jieba.cut(text))
    src_ids = vocab_zh.encode(src_tokens)
    src = torch.tensor([src_ids], device=device)

    # Set Chinese font support
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK']
    plt.rcParams['axes.unicode_minus'] = False

    # Create multi-head attention visualization
    fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=(4 * ((num_heads + 1) // 2), 8))
    axes = axes.flatten()

    for head_idx in range(num_heads):
        attention_weights = get_transformer_attention_weights(model, src, layer_idx, head_idx)

        if attention_weights is not None:
            weights = attention_weights[:len(src_tokens), :len(src_tokens)].numpy()
            im = axes[head_idx].imshow(weights, cmap='Blues', aspect='auto')
            axes[head_idx].set_title(f'Head {head_idx}', fontsize=10)
            axes[head_idx].set_xticks(range(len(src_tokens)))
            axes[head_idx].set_yticks(range(len(src_tokens)))
            axes[head_idx].set_xticklabels(src_tokens, rotation=45, ha='right', fontsize=8)
            axes[head_idx].set_yticklabels(src_tokens, fontsize=8)

    # Hide extra subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'{title_prefix} Multi-Head Attention (Layer {layer_idx})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'transformer_multihead_layer{layer_idx}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Multi-head attention saved to: {output_path}")
    plt.close()


if __name__ == '__main__':
    import json

    # Generate all training curves
    generate_all_curves()

    # Generate attention visualizations
    print("\n=== Generating Attention Visualizations ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load vocabularies
    try:
        from data.scripts.prepare_data import load_vocabs
        vocab_zh, vocab_en = load_vocabs('data/processed_10k')
        print(f"Vocab loaded: zh={len(vocab_zh)}, en={len(vocab_en)}")
    except Exception as e:
        print(f"Failed to load vocab: {e}")
        vocab_zh, vocab_en = None, None

    if vocab_zh and vocab_en:
        test_text = "今天天气很好"  # "The weather is nice today" in Chinese
        print(f"Test text: {test_text}")

        # RNN Attention
        rnn_checkpoint = 'checkpoints/rnn/best_general_10k.pt'
        if os.path.exists(rnn_checkpoint):
            try:
                from models import build_model
                ckpt = torch.load(rnn_checkpoint, map_location=device)
                config = ckpt.get('config', {})
                config.setdefault('src_vocab_size', len(vocab_zh))
                config.setdefault('tgt_vocab_size', len(vocab_en))

                rnn_model = build_model('rnn', config)
                rnn_model.load_state_dict(ckpt['model_state_dict'])
                rnn_model.to(device).eval()
                print(f"RNN model loaded: {rnn_checkpoint}")

                visualize_rnn_attention(
                    rnn_model, test_text, vocab_zh, vocab_en,
                    'figures/rnn_attention.png',
                    title='RNN Encoder-Decoder Attention',
                    device=device
                )
            except Exception as e:
                print(f"RNN attention visualization failed: {e}")
        else:
            print(f"RNN checkpoint not found: {rnn_checkpoint}")

        # Transformer Attention
        transformer_checkpoint = 'checkpoints/transformer/best_absolute_10k.pt'
        if os.path.exists(transformer_checkpoint):
            try:
                from models import build_model
                ckpt = torch.load(transformer_checkpoint, map_location=device)
                config = ckpt.get('config', {})
                config.setdefault('src_vocab_size', len(vocab_zh))
                config.setdefault('tgt_vocab_size', len(vocab_en))

                transformer_model = build_model('transformer', config)
                transformer_model.load_state_dict(ckpt['model_state_dict'])
                transformer_model.to(device).eval()
                print(f"Transformer model loaded: {transformer_checkpoint}")

                visualize_transformer_attention(
                    transformer_model, test_text, vocab_zh,
                    'figures/transformer_attention.png',
                    title='Transformer Self-Attention',
                    layer_idx=0, head_idx=0,
                    device=device
                )

                visualize_transformer_multihead(
                    transformer_model, test_text, vocab_zh,
                    'figures',
                    title_prefix='Transformer',
                    layer_idx=0,
                    device=device
                )
            except Exception as e:
                print(f"Transformer attention visualization failed: {e}")
        else:
            print(f"Transformer checkpoint not found: {transformer_checkpoint}")

    print("\n=== All visualizations completed ===")
