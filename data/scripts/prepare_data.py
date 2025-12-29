import os
import json
import re
import pickle
from collections import Counter
from typing import List, Dict, Tuple

import torch
import numpy as np  # Import numpy for convenient processing
from torch.utils.data import Dataset, DataLoader
import jieba

# Special Tokens
PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>', '<unk>']

# Default paths
DEFAULT_RAW_DIR = 'data/raw'
DEFAULT_OUTPUT = 'data/processed'


def clean_text(text: str, is_chinese: bool = True) -> str:
    """Text cleaning"""
    text = re.sub(r'\s+', ' ', text).strip()
    if is_chinese:
        # Keep Chinese, English, numbers and common punctuation to avoid over-cleaning
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9.,!?;:\'"()\-\s]', '', text)
    else:
        # English keeps ASCII visible characters
        text = re.sub(r'[^\x20-\x7E]', '', text)
    return text


def tokenize_chinese(text: str) -> List[str]:
    return list(jieba.cut(text))


def tokenize_english(text: str) -> List[str]:
    # Enhanced English tokenization: separate punctuation from words
    text = re.sub(r'([.,!?;:\'"()\-])', r' \1 ', text)
    return [t for t in text.lower().split() if t]


class Vocabulary:
    def __init__(self):
        self.word2idx = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.idx2word = {i: t for i, t in enumerate(SPECIAL_TOKENS)}

    def build(self, texts: List[List[str]], max_size: int = 5000, min_freq: int = 1):
        """
        Build vocabulary
        1. Count all word frequencies
        2. Filter out words below min_freq
        3. Sort by frequency and take top max_size
        """
        counter = Counter(w for tokens in texts for w in tokens)

        # 1. Filter & sort (by frequency descending)
        valid_words = sorted(
            [w for w, f in counter.items() if f >= min_freq],
            key=lambda w: counter[w],
            reverse=True
        )

        # 2. Truncate (reserve space for special tokens)
        vocab_slots = max_size - len(SPECIAL_TOKENS)
        valid_words = valid_words[:vocab_slots]

        # 3. Fill
        for word in valid_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                
        print(f"Vocab built: {len(self)} words (Original unique: {len(counter)})")
        return self

    def encode(self, tokens: List[str], add_sos=False, add_eos=False) -> List[int]:
        ids = [self.word2idx.get(t, UNK) for t in tokens]
        if add_sos: ids = [SOS] + ids
        if add_eos: ids = ids + [EOS]
        return ids

    def decode(self, ids: List[int]) -> List[str]:
        # Skip special symbols when decoding
        return [self.idx2word.get(i, '<unk>') for i in ids if i not in (PAD, SOS, EOS)]

    def __len__(self): return len(self.word2idx)

    def save(self, path):
        with open(path, 'wb') as f: pickle.dump(self.word2idx, f)

    @classmethod
    def load(cls, path):
        v = cls()
        with open(path, 'rb') as f: v.word2idx = pickle.load(f)
        v.idx2word = {i: w for w, i in v.word2idx.items()}
        return v


class NMTDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src, self.tgt = src_data, tgt_data

    def __len__(self): return len(self.src)

    def __getitem__(self, i):
        return torch.tensor(self.src[i]), torch.tensor(self.tgt[i])


def collate_fn(batch):
    src, tgt = zip(*batch)
    src_lens = [len(s) for s in src]
    tgt_lens = [len(t) for t in tgt]

    # Pad src and tgt
    src_pad = torch.zeros(len(batch), max(src_lens), dtype=torch.long)
    tgt_pad = torch.zeros(len(batch), max(tgt_lens), dtype=torch.long)

    for i, (s, t) in enumerate(zip(src, tgt)):
        src_pad[i, :len(s)] = s
        tgt_pad[i, :len(t)] = t

    return {
        'src': src_pad,
        'tgt': tgt_pad,
        'src_lengths': torch.tensor(src_lens) # RNN uses pack_padded_sequence which needs lengths
    }


def get_dataloader(data_path: str, batch_size: int = 32, shuffle: bool = True):
    """获取DataLoader"""
    data = torch.load(data_path)
    return DataLoader(NMTDataset(data['src'], data['tgt']),
                      batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(file_path, encoding='utf-8') as f:
        for l in f:
            if l.strip():
                try:
                    data.append(json.loads(l))
                except:
                    pass
    return data


def tokenize_data(data: List[Dict]) -> Tuple[List[List[str]], List[List[str]]]:
    """Tokenize data"""
    zh_texts = [tokenize_chinese(clean_text(d['zh'], True)) for d in data]
    en_texts = [tokenize_english(clean_text(d['en'], False)) for d in data]
    return zh_texts, en_texts


def encode_data(zh_texts: List[List[str]], en_texts: List[List[str]],
                vocab_zh: Vocabulary, vocab_en: Vocabulary,
                max_length: int) -> Tuple[List, List]:
    """Encode and filter data"""
    src_enc, tgt_enc = [], []
    for zh, en in zip(zh_texts, en_texts):
        if len(zh) == 0 or len(en) == 0:
            continue
        src_ids = vocab_zh.encode(zh)
        tgt_ids = vocab_en.encode(en, add_sos=True, add_eos=True)
        if len(src_ids) <= max_length and len(tgt_ids) <= max_length:
            src_enc.append(src_ids)
            tgt_enc.append(tgt_ids)
    return src_enc, tgt_enc


def prepare_data(raw_dir: str = DEFAULT_RAW_DIR, output_dir: str = DEFAULT_OUTPUT,
                 data_size: str = '10k', vocab_size: int = 5000, max_length: int = 128,
                 min_freq: int = 1, embed_path_zh: str = None, embed_path_en: str = None,
                 embed_dim: int = 300, train_embeddings: bool = False):
    """
    Process dataset, generate train.pt, valid.pt
    Vocabulary is built only from training data
    Test set directly uses data/raw/test.jsonl (no preprocessing needed)

    Args:
        data_size: '10k' or '100k', determines which training set to use
        output_dir: Output directory, recommended to use data/processed_10k or data/processed_100k
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine data file paths
    train_file = os.path.join(raw_dir, f'train_{data_size}.jsonl')
    valid_file = os.path.join(raw_dir, 'valid.jsonl')

    # 1. Load data
    print(f"Loading training data from {train_file}...")
    train_data = load_jsonl(train_file)
    print(f"Loading validation data from {valid_file}...")
    valid_data = load_jsonl(valid_file)

    print(f"Data sizes: train={len(train_data)}, valid={len(valid_data)}")

    # 2. Tokenize
    print("Tokenizing...")
    train_zh, train_en = tokenize_data(train_data)
    valid_zh, valid_en = tokenize_data(valid_data)

    # 3. Build vocabularies from training data only
    print("Building vocabularies (from training data only)...")
    vocab_zh = Vocabulary().build(train_zh, vocab_size, min_freq)
    vocab_en = Vocabulary().build(train_en, vocab_size, min_freq)

    vocab_zh.save(os.path.join(output_dir, 'vocab_zh.pkl'))
    vocab_en.save(os.path.join(output_dir, 'vocab_en.pkl'))

    # 4. Word embedding processing
    has_embeddings = False

    def handle_embedding(embed_path, vocab, texts, save_name):
        emb = None
        if embed_path and os.path.exists(embed_path):
            emb = load_pretrained_embeddings(embed_path, vocab, embed_dim)
        elif train_embeddings:
            emb = train_word2vec(texts, vocab, embed_dim)

        if emb is not None:
            torch.save(emb, os.path.join(output_dir, save_name))
            return True
        return False

    has_zh = handle_embedding(embed_path_zh, vocab_zh, train_zh, 'embeddings_zh.pt')
    has_en = handle_embedding(embed_path_en, vocab_en, train_en, 'embeddings_en.pt')
    has_embeddings = has_zh or has_en

    # 5. Encode datasets
    print("Encoding datasets...")

    train_src, train_tgt = encode_data(train_zh, train_en, vocab_zh, vocab_en, max_length)
    print(f"  Train: {len(train_src)} samples")

    valid_src, valid_tgt = encode_data(valid_zh, valid_en, vocab_zh, vocab_en, max_length)
    print(f"  Valid: {len(valid_src)} samples")

    # 6. Save datasets
    torch.save({'src': train_src, 'tgt': train_tgt}, os.path.join(output_dir, 'train.pt'))
    torch.save({'src': valid_src, 'tgt': valid_tgt}, os.path.join(output_dir, 'valid.pt'))
    torch.save({'src': train_src, 'tgt': train_tgt}, os.path.join(output_dir, 'data.pt'))

    print(f"Data saved to {output_dir}/")
    print(f"Note: Use data/raw/test.jsonl for evaluation (no preprocessing needed)")

    # Save configuration
    config = {
        'src_vocab_size': len(vocab_zh),
        'tgt_vocab_size': len(vocab_en),
        'pad_idx': PAD,
        'sos_idx': SOS,
        'eos_idx': EOS,
        'unk_idx': UNK,
        'max_length': max_length,
        'embed_dim': embed_dim if has_embeddings else None,
        'data_size': data_size,
        'train_samples': len(train_src),
        'valid_samples': len(valid_src)
    }
    json.dump(config, open(os.path.join(output_dir, 'config.json'), 'w'), indent=2)
    print("Config saved.")


def load_pretrained_embeddings(embed_path: str, vocab: Vocabulary, embed_dim: int = 300):
    print(f"Loading pretrained embeddings from {embed_path}...")
    # Use Xavier initialization / normal distribution by default instead of all zeros
    embeddings = torch.randn(len(vocab), embed_dim) * 0.1 
    found = 0

    with open(embed_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            if len(parts) <= 2: continue
            word = parts[0]
            vec = parts[1:]
            if word in vocab.word2idx and len(vec) == embed_dim:
                idx = vocab.word2idx[word]
                embeddings[idx] = torch.tensor([float(v) for v in vec])
                found += 1

    # Only PAD must be 0, SOS/EOS/UNK should keep random initialization (with gradients)
    embeddings[PAD] = 0

    print(f"Found {found}/{len(vocab)} words.")
    return embeddings


def train_word2vec(texts: List[List[str]], vocab: Vocabulary, embed_dim: int = 300,
                   window: int = 5, min_count: int = 1, epochs: int = 10):
    from gensim.models import Word2Vec
    print(f"Training Word2Vec...")
    # Note: min_count here must be consistent or smaller than vocab's min_freq
    model = Word2Vec(sentences=texts, vector_size=embed_dim, window=window,
                     min_count=min_count, workers=4, epochs=epochs)

    embeddings = torch.randn(len(vocab), embed_dim) * 0.1 # Random initialization for uncovered words
    found = 0
    for word, idx in vocab.word2idx.items():
        if word in model.wv:
            embeddings[idx] = torch.tensor(model.wv[word])
            found += 1

    embeddings[PAD] = 0 # Only PAD is set to 0
    print(f"Word2Vec coverage: {found}/{len(vocab)}")
    return embeddings

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', default=DEFAULT_RAW_DIR, help='Raw data directory')
    parser.add_argument('--output', default=None, help='Output directory (default: data/processed_{data})')
    parser.add_argument('--data', default='10k', choices=['10k', '100k'],
                        help='Dataset size: 10k or 100k')
    parser.add_argument('--vocab_size', type=int, default=5000)
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--min_freq', type=int, default=1)
    parser.add_argument('--embed_path_zh', default=None)
    parser.add_argument('--embed_path_en', default=None)
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--train_embeddings', action='store_true')

    args = parser.parse_args()

    # Default output directory is distinguished by dataset
    output_dir = args.output if args.output else f'data/processed_{args.data}'

    prepare_data(
        raw_dir=args.raw_dir,
        output_dir=output_dir,
        data_size=args.data,
        vocab_size=args.vocab_size,
        max_length=args.max_length,
        min_freq=args.min_freq,
        embed_path_zh=args.embed_path_zh,
        embed_path_en=args.embed_path_en,
        embed_dim=args.embed_dim,
        train_embeddings=args.train_embeddings
    )