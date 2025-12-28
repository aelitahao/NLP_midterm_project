# NLP_midterm_project
Machine Translation between Chinese and English

## Install Dependencies

```bash
pip install torch pyyaml jieba gensim transformers
```

## Training

### RNN/Transformer Model (train.py)

```bash
python train.py --model_type <rnn|transformer> --data_dir <path> [options]
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_type` | transformer | Model type: `rnn` or `transformer` |
| `--data_dir` | data/processed | Preprocessed data directory |
| `--name` | None | Checkpoint name suffix (e.g., `general_10k`) |
| `--resume` | None | Resume from checkpoint path |
| `--batch_size` | config | Batch size |
| `--learning_rate` | config | Learning rate |
| `--device` | cuda | Device: `cuda` or `cpu` |

**RNN-specific:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--attention_type` | general | Attention: `dot`, `general`, `additive` |
| `--teacher_forcing_ratio` | 1.0 | TF ratio (1.0=TF, 0.0=Free Running) |
| `--tf_decay` | False | Enable TF ratio decay |

**Transformer-specific:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pos_encoding` | absolute | Position encoding: `absolute`, `relative` |
| `--use_rmsnorm` | False | Use RMSNorm instead of LayerNorm |
| `--d_model` | config | Model dimension |
| `--num_heads` | config | Number of attention heads |
| `--num_layers` | config | Number of layers |
| `--d_ff` | config | Feed-forward dimension |

**Examples:**

```bash
# RNN with general attention
python train.py --model_type rnn --data_dir data/processed_10k --attention_type general --name general_10k

# Transformer with relative position encoding
python train.py --model_type transformer --data_dir data/processed_100k --pos_encoding relative --name relative_100k
```

### T5 Model (train_t5.py)

```bash
python train_t5.py --data <10k|100k> [options]
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | (required) | Dataset: `10k` or `100k` |
| `--model` | t5-small | T5 model: `t5-small`, `t5-base`, `t5-large`, `t5-3b` |
| `--epochs` | 10 | Number of epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 3e-4 | Learning rate |
| `--max_len` | 128 | Max sequence length |
| `--name` | None | Checkpoint name suffix |
| `--use_lora` | False | Enable LoRA fine-tuning |
| `--grad_accum` | 1 | Gradient accumulation steps |
| `--amp` | none | Mixed precision: `none`, `fp16`, `bf16` |

**Examples:**

```bash
# T5-small on 10k dataset
python train_t5.py --data 10k --epochs 10

# T5-base on 100k with mixed precision
python train_t5.py --data 100k --model t5-base --batch_size 16 --amp bf16 --name base_100k
```

## Inference

```bash
python inference.py --checkpoint <path> [options]
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | (required) | Model checkpoint path |
| `--data_dir` | data/processed | Data directory (for RNN/Transformer) |
| `--text` | None | Single sentence to translate |
| `--input` | None | Input file (JSONL format) |
| `--output` | translations.txt | Output file |
| `--evaluate` | False | Compute BLEU score |
| `--interactive` | False | Interactive mode |
| `--beam_width` | 5 | Beam search width |
| `--t5` | False | Use T5 model |
| `--t5_model` | t5-small | T5 pretrained model name |

**Examples:**

```bash
# Single sentence (RNN/Transformer)
python inference.py --checkpoint checkpoints/rnn/best_100k.pt --data_dir data/processed_100k --text "今天天气很好"

# BLEU evaluation
python inference.py --checkpoint checkpoints/transformer/best_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate

# Interactive mode
python inference.py --checkpoint checkpoints/transformer/best_100k.pt --data_dir data/processed_100k --interactive

# T5 model inference
python inference.py --t5 --checkpoint checkpoints/t5/latest_10k.pt --text "今天天气很好"

# T5 BLEU evaluation
python inference.py --t5 --t5_model t5-base --checkpoint checkpoints/t5/latest_base_100k.pt --input data/raw/test.jsonl --evaluate
```
