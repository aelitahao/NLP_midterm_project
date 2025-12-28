import os
import json
import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from transformers import T5ForConditionalGeneration, T5Tokenizer

try:
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel
    PEFT_INSTALLED = True
except ImportError:
    PEFT_INSTALLED = False
    print("Warning: 'peft' library not installed. LoRA training will not be available.")

class T5Dataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        # 1. Preload all data into memory
        print(f"Loading data from {path}...")
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 2. Keep dynamic tokenization with high num_workers
        src = "translate Chinese to English: " + self.data[idx]['zh']
        tgt = self.data[idx]['en']

        src_enc = self.tokenizer(src, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        tgt_enc = self.tokenizer(tgt, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')

        labels = tgt_enc['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': src_enc['input_ids'].squeeze(),
            'attention_mask': src_enc['attention_mask'].squeeze(),
            'labels': labels
        }

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, default='t5-small')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--use_lora', type=str, default='False', help="Enable LoRA (True/False)")
    parser.add_argument('--grad_accum', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument('--workers', type=int, default=8, help="DataLoader workers")
    parser.add_argument('--amp', type=str, default='none', choices=['none', 'fp16', 'bf16'],
                        help="Mixed precision: none (default, stable), fp16, bf16 (recommended if supported)")

    args = parser.parse_args()
    args.use_lora = args.use_lora.lower() == 'true'

    # Path configuration
    train_path = f"data/raw/train_{args.data}.jsonl"
    valid_path = "data/raw/valid.jsonl"
    save_dir = "checkpoints/t5"
    os.makedirs(save_dir, exist_ok=True)
    name = args.name or args.data

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Initializing {args.model} on {device}...")

    # Model loading
    tokenizer = T5Tokenizer.from_pretrained(args.model)

    # Memory optimization: For large models, don't load to GPU until LoRA wrapper is ready
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    if args.use_lora:
        if not PEFT_INSTALLED:
            raise ImportError("Please install peft: pip install peft")

        print("Enabling LoRA Fine-tuning...")
        # LoRA config for T5
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,            # LoRA rank, higher means more parameters
            lora_alpha=32,
            lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters() # Print parameter count comparison
    else:
        print("Running full-parameter fine-tuning.")

    model = model.to(device)

    # Data
    train_set = T5Dataset(train_path, tokenizer, args.max_len)
    valid_set = T5Dataset(valid_path, tokenizer, args.max_len)
    
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, 
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set, 
        batch_size=args.batch_size, 
        num_workers=args.workers, 
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # AMP configuration
    use_amp = args.amp != 'none'
    amp_dtype = torch.bfloat16 if args.amp == 'bf16' else torch.float16
    scaler = GradScaler('cuda') if args.amp == 'fp16' else None  # bf16 doesn't need scaler

    best_loss = float('inf')

    # Print actual batch size
    effective_bs = args.batch_size * args.grad_accum
    print(f"\nStart Training: {args.data} | Effective Batch Size: {effective_bs} | AMP: {args.amp}")

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0
        optimizer.zero_grad(set_to_none=True) # Memory micro-optimization

        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if use_amp:
                with autocast(device_type='cuda', dtype=amp_dtype):
                    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                    loss = loss / args.grad_accum
                if scaler:  # fp16
                    scaler.scale(loss).backward()
                else:  # bf16
                    loss.backward()
            else:
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                loss = loss / args.grad_accum
                loss.backward()

            # Only update parameters when accumulation steps are reached
            if (i + 1) % args.grad_accum == 0:
                if scaler:  # fp16
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * args.grad_accum # Restore loss value for printing

            if i % 50 == 0:
                print(f"Epoch {epoch} | Batch {i}/{len(train_loader)} | Loss: {loss.item() * args.grad_accum:.4f}")

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                if use_amp:
                    with autocast(device_type='cuda', dtype=amp_dtype):
                        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                else:
                    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                val_loss += loss.item()
        val_loss /= len(valid_loader)

        print(f"Epoch {epoch} | Time: {time.time()-t0:.1f}s | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # If LoRA, save_pretrained only saves adapter, very small
        # If full fine-tuning, state_dict saves everything
        save_path_latest = f"{save_dir}/latest_{name}"
        save_path_best = f"{save_dir}/best_{name}"
        
        if args.use_lora:
            model.save_pretrained(save_path_latest)
        else:
            torch.save(model.state_dict(), save_path_latest + ".pt")

        if val_loss < best_loss:
            best_loss = val_loss
            if args.use_lora:
                model.save_pretrained(save_path_best)
            else:
                torch.save(model.state_dict(), save_path_best + ".pt")
            print(f"New best model saved! Val Loss: {val_loss:.4f}")

    print(f"\nDone! Best Val Loss: {best_loss:.4f}")

if __name__ == '__main__':
    train()