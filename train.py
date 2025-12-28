"""Training main program (RNN / Transformer)"""

import os
import json
import argparse
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from models import build_model
from data.scripts.prepare_data import get_dataloader, NMTDataset, collate_fn
from torch.utils.data import DataLoader, random_split


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Weight decay for regularization (mainly for RNN)
        weight_decay = config.get('weight_decay', 0)
        self.optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-4), weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.get('pad_idx', 0))

        self.use_amp = config.get('use_amp', True) and device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None

        # Warmup scheduler for Transformer
        self.warmup_steps = config.get('warmup_steps', 0)
        self.current_step = 0
        self.base_lr = config.get('learning_rate', 1e-4)

        # Save checkpoints by model type in separate directories
        base_dir = config.get('checkpoint_dir', 'checkpoints')
        model_type = config.get('model_type', 'transformer')
        self.checkpoint_dir = os.path.join(base_dir, model_type)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = float('inf')

        # Checkpoint naming suffix (if --name parameter is specified)
        self.checkpoint_name = config.get('checkpoint_name', None)

    # def _update_lr(self):
    #     """Warmup learning rate scheduler"""
    #     if self.warmup_steps > 0:
    #         self.current_step += 1
    #         if self.current_step < self.warmup_steps:
    #             lr = self.base_lr * self.current_step / self.warmup_steps
    #         else:
    #             lr = self.base_lr
    #         for param_group in self.optimizer.param_groups:
    #             param_group['lr'] = lr

    def _update_lr(self):
        '''Transformer standard scheduler: Warmup + Inverse Square Root Decay'''
        self.current_step += 1

        # Avoid division by zero when step=0 (although usually starts from 1, but for robustness)
        step = max(1, self.current_step)

        if self.warmup_steps > 0:
            if step < self.warmup_steps:
                # Warmup phase: linearly increase
                # lr = base_lr * (step / warmup)
                lr = self.base_lr * (step / self.warmup_steps)
            else:
                # Decay phase: inverse square root decay
                # Here the logic is: at step == warmup, decay factor is 1.0 (seamless transition)
                # As step increases, learning rate decreases by the inverse square root of step
                decay_factor = (self.warmup_steps / step) ** 0.5
                lr = self.base_lr * decay_factor
        else:
            # If no warmup (not recommended), use constant lr
            lr = self.base_lr

        # Update learning rate in optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train_epoch(self, epoch, tf_ratio=1.0):
        self.model.train()
        total_loss = 0
        model_type = self.config.get('model_type', 'transformer')

        for batch_idx, batch in enumerate(self.train_loader):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            src_lengths = batch.get('src_lengths')
            if src_lengths is not None:
                src_lengths = src_lengths.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                if model_type == 'rnn':
                    outputs = self.model(src, tgt, src_lengths, tf_ratio)
                else:
                    outputs = self.model(src, tgt[:, :-1])
                loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            # Update learning rate (warmup)
            self._update_lr()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | Loss: {loss.item():.4f}", flush=True)

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        model_type = self.config.get('model_type', 'transformer')

        for batch in self.val_loader:
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            src_lengths = batch.get('src_lengths')
            if src_lengths is not None:
                src_lengths = src_lengths.to(self.device)

            with autocast(enabled=self.use_amp):
                if model_type == 'rnn':
                    outputs = self.model(src, tgt, src_lengths, 1.0)
                else:
                    outputs = self.model(src, tgt[:, :-1])
                loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))

            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, val_loss, save_every=1):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }

        # Decide filename based on whether --name is specified
        if self.checkpoint_name:
            latest_name = f'latest_{self.checkpoint_name}.pt'
            best_name = f'best_{self.checkpoint_name}.pt'
            epoch_name = f'epoch_{epoch}_{self.checkpoint_name}.pt'
        else:
            latest_name = 'latest.pt'
            best_name = 'best.pt'
            epoch_name = f'epoch_{epoch}.pt'

        # Always save latest
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, latest_name))

        # Save epoch checkpoint at save_every interval
        if (epoch + 1) % save_every == 0:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, epoch_name))

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, best_name))
            print(f"New best model! Val Loss: {val_loss:.4f}", flush=True)

    def train(self, epochs, resume_from=None):
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1

        print(f"Training {self.config.get('model_type', 'transformer')} on {self.device}", flush=True)
        if self.config.get('model_type') == 'rnn':
            print(f"Teacher forcing ratio: {self.config.get('teacher_forcing_ratio', 1.0)} (decay: {self.config.get('tf_decay', False)})", flush=True)

        for epoch in range(start_epoch, epochs):
            start_time = time.time()

            # Teacher forcing ratio for RNN (from config, with optional decay)
            if self.config.get('model_type') == 'rnn':
                base_tf = self.config.get('teacher_forcing_ratio', 1.0)
                if self.config.get('tf_decay', False):
                    # Optional decay: from base_tf to base_tf*0.5 over 15 epochs
                    tf_ratio = max(base_tf * 0.5, base_tf - epoch / 15)
                else:
                    tf_ratio = base_tf
            else:
                tf_ratio = 1.0

            train_loss = self.train_epoch(epoch, tf_ratio)
            val_loss = self.validate()

            print(f"Epoch {epoch} | Time: {time.time()-start_time:.1f}s | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)

            save_every = self.config.get('save_every', 1)
            self.save_checkpoint(epoch, val_loss, save_every=save_every)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--model_type', default='transformer', choices=['rnn', 'transformer'])
    parser.add_argument('--data_dir', default='data/processed')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--name', default=None,
                        help='Checkpoint name suffix (e.g., dot_10k -> best_dot_10k.pt). If not set, uses best.pt')
    # RNN specific parameters
    parser.add_argument('--attention_type', default='general', choices=['dot', 'general', 'additive'],
                        help='Attention type for RNN model')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0,
                        help='Teacher forcing ratio (1.0=TF, 0.0=Free Running)')
    parser.add_argument('--tf_decay', action='store_true',
                        help='Enable teacher forcing decay')
    # Transformer specific parameters
    parser.add_argument('--pos_encoding', default='absolute', choices=['absolute', 'relative'],
                        help='Position encoding type for Transformer')
    parser.add_argument('--use_rmsnorm', action='store_true',
                        help='Use RMSNorm instead of LayerNorm')
    parser.add_argument('--d_model', type=int, default=None,
                        help='Model dimension (overrides config)')
    parser.add_argument('--num_heads', type=int, default=None,
                        help='Number of attention heads (overrides config)')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Number of layers (overrides config)')
    parser.add_argument('--d_ff', type=int, default=None,
                        help='Feed-forward dimension (overrides config)')
    # Common training parameters
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Merge model-specific config
    model_config = config.get(args.model_type, {})
    config.update(model_config)
    config['model_type'] = args.model_type

    # Command line arguments override config (RNN related)
    if args.model_type == 'rnn':
        config['attention_type'] = args.attention_type
        config['teacher_forcing_ratio'] = args.teacher_forcing_ratio
        config['tf_decay'] = args.tf_decay

    # Command line arguments override config (Transformer related)
    if args.model_type == 'transformer':
        config['pos_encoding'] = args.pos_encoding
        config['use_rmsnorm'] = args.use_rmsnorm
        if args.d_model is not None:
            config['d_model'] = args.d_model
        if args.num_heads is not None:
            config['num_heads'] = args.num_heads
        if args.num_layers is not None:
            config['num_layers'] = args.num_layers
        if args.d_ff is not None:
            config['d_ff'] = args.d_ff

    # Common parameter overrides
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.name is not None:
        config['checkpoint_name'] = args.name

    # Load data config
    data_config_path = os.path.join(args.data_dir, 'config.json')
    if os.path.exists(data_config_path):
        with open(data_config_path) as f:
            config.update(json.load(f))
    else:
        config.setdefault('vocab_size', 5000)
        config['pad_idx'] = 0

    config.setdefault('src_vocab_size', config.get('vocab_size', 5000))
    config.setdefault('tgt_vocab_size', config.get('vocab_size', 5000))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load data
    train_path = os.path.join(args.data_dir, 'train.pt')
    valid_path = os.path.join(args.data_dir, 'valid.pt')
    data_path = os.path.join(args.data_dir, 'data.pt')  # Legacy compatibility
    batch_size = config.get('batch_size', 32)

    # Use independent train.pt and valid.pt first
    if os.path.exists(train_path) and os.path.exists(valid_path):
        print(f"Loading train.pt and valid.pt from {args.data_dir}")
        train_data = torch.load(train_path)
        valid_data = torch.load(valid_path)
        train_set = NMTDataset(train_data['src'], train_data['tgt'])
        val_set = NMTDataset(valid_data['src'], valid_data['tgt'])
        print(f"Train samples: {len(train_set)}, Valid samples: {len(val_set)}")
        train_loader = DataLoader(train_set, batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size, shuffle=False, collate_fn=collate_fn)
    elif os.path.exists(data_path):
        # Legacy compatibility: use random_split
        print(f"Warning: Using legacy data.pt with random split (no valid.pt found)")
        data = torch.load(data_path)
        dataset = NMTDataset(data['src'], data['tgt'])
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        print("Data not found. Run prepare_data.py first.")
        return

    # Build model
    model = build_model(args.model_type, config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Train
    trainer = Trainer(model, train_loader, val_loader, config, device)

    total_start_time = time.time()
    trainer.train(config.get('epochs', 20), args.resume)
    total_time = time.time() - total_start_time

    # Output total training time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"\nTotal training time: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_time:.1f}s)", flush=True)


if __name__ == '__main__':
    main()
