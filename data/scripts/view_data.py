"""View preprocessed data"""

import os
import json
import pickle
import argparse
import torch


def view_data(data_dir: str, num_samples: int = 5):
    """View preprocessed data"""

    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print("=" * 60)

    # === 1. View configuration ===
    config_path = os.path.join(data_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print("\n【Configuration】")
        for k, v in config.items():
            print(f"  {k}: {v}")

    # === 2. Load vocabularies ===
    vocab_zh_path = os.path.join(data_dir, 'vocab_zh.pkl')
    vocab_en_path = os.path.join(data_dir, 'vocab_en.pkl')

    with open(vocab_zh_path, 'rb') as f:
        vocab_zh = pickle.load(f)
    with open(vocab_en_path, 'rb') as f:
        vocab_en = pickle.load(f)

    idx2word_zh = {v: k for k, v in vocab_zh.items()}
    idx2word_en = {v: k for k, v in vocab_en.items()}

    print("\n【Vocabulary Info】")
    print(f"  Chinese vocab size: {len(vocab_zh)}")
    print(f"  English vocab size: {len(vocab_en)}")
    print(f"  Chinese vocab first 20 words: {list(vocab_zh.keys())[:20]}")
    print(f"  English vocab first 20 words: {list(vocab_en.keys())[:20]}")

    # === 3. Load data ===
    data_path = os.path.join(data_dir, 'data.pt')
    data = torch.load(data_path)

    src_data = data['src']
    tgt_data = data['tgt']

    # === 4. Statistics ===
    src_lens = [len(s) for s in src_data]
    tgt_lens = [len(t) for t in tgt_data]

    print("\n【Data Statistics】")
    print(f"  Number of samples: {len(src_data)}")
    print(f"  Source sentence length: avg={sum(src_lens)/len(src_lens):.1f}, max={max(src_lens)}, min={min(src_lens)}")
    print(f"  Target sentence length: avg={sum(tgt_lens)/len(tgt_lens):.1f}, max={max(tgt_lens)}, min={min(tgt_lens)}")

    # === 5. Sample display ===
    print(f"\n【Sample Display】(first {num_samples} samples)")
    print("-" * 60)

    for i in range(min(num_samples, len(src_data))):
        src_ids = src_data[i]
        tgt_ids = tgt_data[i]

        # Decode (skip special tokens: pad=0, sos=1, eos=2)
        src_tokens = [idx2word_zh.get(idx, '<unk>') for idx in src_ids if idx not in (0, 1, 2)]
        tgt_tokens = [idx2word_en.get(idx, '<unk>') for idx in tgt_ids if idx not in (0, 1, 2)]

        src_text = ''.join(src_tokens)
        tgt_text = ' '.join(tgt_tokens)

        print(f"\nSample {i+1}:")
        print(f"  Source (Chinese): {src_text}")
        print(f"  Target (English): {tgt_text}")
        print(f"  Source IDs: {src_ids[:10]}{'...' if len(src_ids) > 10 else ''}")
        print(f"  Target IDs: {tgt_ids[:10]}{'...' if len(tgt_ids) > 10 else ''}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='View preprocessed data')
    parser.add_argument('--data_dir', default='../processed_10k', help='Data directory')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to display')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        print("Please run data preprocessing first:")
        print("  python prepare_data.py --input ../raw/train_10k.jsonl --output ../processed_10k")
        return

    view_data(args.data_dir, args.num_samples)


if __name__ == '__main__':
    main()
