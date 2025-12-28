"""One-click inference script"""

import os
import json
import argparse
import time
import torch
from models import build_model, greedy_decode, beam_search
from utils import BLEUScorer

try:
    from data.scripts.prepare_data import load_vocabs, tokenize_chinese, tokenize_english, Vocabulary
except:
    load_vocabs = None

try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    HAS_T5 = True
except ImportError:
    HAS_T5 = False

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


class T5Inference:
    """T5 model inference class, supports full fine-tuning and LoRA"""
    def __init__(self, checkpoint_path, model_name='t5-small', device='cuda'):
        if not HAS_T5:
            raise ImportError("T5 inference requires transformers library: pip install transformers")

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_name = os.path.basename(checkpoint_path)
        self.model_type = 't5'

        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Determine if LoRA checkpoint (directory) or full fine-tuning (.pt file)
        if os.path.isdir(checkpoint_path):
            # LoRA checkpoint
            if not HAS_PEFT:
                raise ImportError("LoRA inference requires peft library: pip install peft")
            print(f"Loading LoRA checkpoint from {checkpoint_path}")
            base_model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
            self.model = self.model.merge_and_unload()
        else:
            # Full fine-tuning checkpoint (.pt file)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device).eval()
        self.task_prefix = "translate Chinese to English: "

    @torch.no_grad()
    def translate(self, text, beam_width=4, max_len=128):
        input_text = self.task_prefix + text
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_len,
            truncation=True
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=max_len,
            num_beams=beam_width,
            early_stopping=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class NMTInference:
    def __init__(self, checkpoint_path, data_dir=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_name = os.path.basename(checkpoint_path)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.config = ckpt.get('config', {})

        # Load data config
        if data_dir:
            config_path = os.path.join(data_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path) as f:
                    self.config.update(json.load(f))

        self.config.setdefault('src_vocab_size', self.config.get('vocab_size', 5000))
        self.config.setdefault('tgt_vocab_size', self.config.get('vocab_size', 5000))

        self.model_type = self.config.get('model_type', 'transformer')
        self.model = build_model(self.model_type, self.config)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device).eval()

        # Load separate vocabularies
        self.vocab_zh = None
        self.vocab_en = None
        if load_vocabs and data_dir:
            try:
                self.vocab_zh, self.vocab_en = load_vocabs(data_dir)
            except:
                pass

    @torch.no_grad()
    def translate(self, text, beam_width=5, max_len=100):
        # Encode (tokenize then convert to IDs)
        if self.vocab_zh:
            tokens = tokenize_chinese(text)
            ids = self.vocab_zh.encode(tokens)
        else:
            ids = [ord(c) % 5000 for c in text]
        src = torch.tensor([ids], device=self.device)

        # Decode
        sos, eos = self.config.get('sos_idx', 1), self.config.get('eos_idx', 2)
        if beam_width > 1:
            out_ids = beam_search(self.model, src, None, beam_width, max_len, sos, eos, self.model_type)
        else:
            out = greedy_decode(self.model, src, None, max_len, sos, eos, self.model_type)
            out_ids = out[0].tolist()

        # Filter special tokens and decode
        filtered = [i for i in out_ids if i not in [0, 1, 2]]
        if self.vocab_en:
            words = self.vocab_en.decode(filtered)
            return ' '.join(words)
        return ''.join(chr(i) for i in filtered if 32 <= i < 127)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_dir', default='data/processed')
    parser.add_argument('--text', default=None)
    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default='translations.txt')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--device', default='cuda')
    # T5 specific parameters
    parser.add_argument('--t5', action='store_true', help='Use T5 model for inference')
    parser.add_argument('--t5_model', default='t5-small', help='T5 pretrained model name')
    args = parser.parse_args()

    # Select inference class based on --t5 parameter
    if args.t5:
        nmt = T5Inference(args.checkpoint, args.t5_model, args.device)
    else:
        nmt = NMTInference(args.checkpoint, args.data_dir, args.device)
    print(f"Model: {nmt.model_type} | Checkpoint: {nmt.checkpoint_name} | Device: {nmt.device}")

    # Single sentence
    if args.text:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        result = nmt.translate(args.text, args.beam_width)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        latency = (time.perf_counter() - start_time) * 1000
        print(f"Input: {args.text}\nTranslation: {result}\nLatency: {latency:.2f} ms")
        return

    # Interactive mode
    if args.interactive or not args.input:
        print("Interactive mode (type 'q' to quit)")
        while True:
            text = input("Input: ").strip()
            if text.lower() in ['q', 'quit', 'exit']:
                break
            if text:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.perf_counter()
                result = nmt.translate(text, args.beam_width)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                latency = (time.perf_counter() - start_time) * 1000
                print(f"Translation: {result}\nLatency: {latency:.2f} ms\n")
        return

    # File mode
    with open(args.input) as f:
        data = [json.loads(l) for l in f if l.strip()]

    if args.evaluate:
        scorer = BLEUScorer()
        latencies = []

        print(f"Evaluating on {len(data)} samples...")
        for i, d in enumerate(data):
            # Measure inference latency
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()

            hyp = nmt.translate(d['zh'], args.beam_width)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()

            latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds
            scorer.add(d['en'], hyp)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(data)}")

        m = scorer.compute()
        avg_latency = sum(latencies) / len(latencies)
        total_time = sum(latencies) / 1000  # Total time in seconds

        print(f"\n{'='*50}")
        print(f"Model: {nmt.model_type} | Checkpoint: {nmt.checkpoint_name}")
        print(f"{'='*50}")
        print(f"BLEU:   {m['bleu']:.2f}")
        print(f"BLEU-1: {m['bleu-1']:.2f}")
        print(f"BLEU-2: {m['bleu-2']:.2f}")
        print(f"BLEU-3: {m['bleu-3']:.2f}")
        print(f"BLEU-4: {m['bleu-4']:.2f}")
        print(f"{'='*50}")
        print(f"Avg Latency: {avg_latency:.2f} ms/sentence")
        print(f"Total Time:  {total_time:.2f} s")
        print(f"Throughput:  {len(data)/total_time:.2f} sentences/s")
        print(f"{'='*50}")
    else:
        trans = [nmt.translate(d.get('zh', ''), args.beam_width) for d in data]
        with open(args.output, 'w') as f:
            f.write('\n'.join(trans))
        print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
