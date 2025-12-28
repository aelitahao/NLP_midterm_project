# 神经机器翻译 (NMT) 对比实验

中英文神经机器翻译系统，支持 RNN Seq2Seq、Transformer 和 T5 模型架构。

## 目录

- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [实验概览](#实验概览)
- [RNN 对比实验](#rnn-对比实验)
- [Transformer 对比实验](#transformer-对比实验)
- [T5 微调实验](#t5-微调实验)
- [模型对比分析](#模型对比分析)
- [实验记录表格](#实验记录表格)

---

## 项目结构

```
nmt_project/
├── models/                     # 模型架构
│   ├── modules.py              # 注意力/归一化/位置编码
│   ├── rnn.py                  # RNN Seq2Seq + Attention
│   ├── transformer.py          # Transformer
│   └── t5_wrapper.py           # T5 微调
├── data/
│   ├── raw/                    # 原始数据 (JSONL)
│   ├── processed/              # 预处理数据
│   └── scripts/prepare_data.py
├── checkpoints/                # 模型检查点（按类型分目录）
│   ├── rnn/
│   ├── transformer/
│   └── t5/
├── log/                        # 训练和评估日志
├── configs/config.yaml         # 配置文件
├── train.py                    # 训练
├── inference.py                # 推理
└── utils.py                    # BLEU评估/解码策略
```

---

## 环境配置

```bash
# 激活conda环境
conda activate nmt_env

# 安装依赖
pip install torch pyyaml jieba gensim
pip install transformers  # 仅T5需要

# 创建日志目录
mkdir -p log
```

---

## 数据准备

分别准备 10k 和 100k 两个数据集：

```bash
# 10k 数据集 (词表 5000)
python data/scripts/prepare_data2.py --data 10k

# 100k 数据集 (词表 10000, min_freq=2 过滤低频词)
python data/scripts/prepare_data2.py --data 100k --vocab_size 10000 --min_freq 2 --max_length 100
```

**词表大小建议：**
| 数据集 | 建议词表大小 | min_freq | max_length |
|--------|-------------|----------|------------|
| 10k | 5,000 | 1 | 64 |
| 100k | 10,000 | 2 | 100 |

生成文件结构（data/processed_10k/ 或 data/processed_100k/）：
- `train.pt`: 训练集
- `valid.pt`: 验证集（来自 data/raw/valid.jsonl）
- `data.pt`: 兼容旧版（= train.pt）
- `vocab_zh.pkl`: 中文词表
- `vocab_en.pkl`: 英文词表
- `config.json`: 配置信息

**注意**：测试集使用 `data/raw/test.jsonl` 直接评估，无需预处理。

---

## 实验概览

### 数据集说明

本实验使用两种规模的数据集进行对比：

| 数据集 | 样本数 | 用途 | 预计训练时间 |
|--------|--------|------|--------------|
| 10k | 10,000 | 快速验证、调试 | ~5小时（全部实验） |
| 100k | 100,000 | 正式实验、论文结果 | ~40小时（全部实验） |

### 实验总数

每个实验需在 **10k 和 100k** 数据集上各运行一次：

| 类别 | 实验内容 | 模型数/数据集 | 总模型数 | 总评估次数 |
|------|----------|--------------|----------|------------|
| RNN | Attention对比 (dot/general/additive) | 3 | 6 | 12 |
| RNN | Training策略对比 (TF/FR) | 1 | 2 | 4 |
| Transformer | Position Encoding对比 | 2 | 4 | 4 |
| Transformer | Normalization对比 | 1 | 2 | 2 |
| Transformer | Batch Size对比 | 2 | 4 | 4 |
| Transformer | Learning Rate对比 | 2 | 4 | 4 |
| Transformer | Model Scale对比 (可选) | 2 | 4 | 4 |
| T5 | Fine-tuning | 1 | 2 | 2 |
| **总计** | | **~14** | **~28** | **~36** |

---

## RNN 对比实验

> **注意**：以下每个实验需分别在 10k 和 100k 数据集上运行

### 实验 R1-R3: Attention 机制对比

比较三种注意力对齐函数：Dot-product、General (Multiplicative)、Additive

#### 步骤 1: 修改配置文件 `configs/config.yaml`

**10k 数据集配置：**
```yaml
rnn:
  embed_size: 256
  hidden_size: 512
  num_layers: 2
  rnn_type: "gru"
  dropout: 0.5                 # 高Dropout防止过拟合
  batch_size: 32
  epochs: 30
  learning_rate: 0.001
```

**100k 数据集配置：**
```yaml
rnn:
  embed_size: 512              # 提升维度，适配 20000 词表
  hidden_size: 512             # 保持与 embed_size 一致或更大
  num_layers: 2                # 2层足够
  rnn_type: "gru"              # GRU 训练效率优于 LSTM
  dropout: 0.3                 # 标准正则化
  batch_size: 64               # 加大 Batch Size 提升训练速度
  epochs: 30
  learning_rate: 0.001
```

> **注意**：`attention_type` 和 `teacher_forcing_ratio` 通过命令行参数指定

#### 步骤 2: 训练（10k 数据集）

```bash
# R1: Dot-product Attention (10k)
nohup python train.py --model_type rnn --data_dir data/processed_10k --attention_type dot --teacher_forcing_ratio 1.0 --name dot_10k > log/rnn_dot_10k_train.log 2>&1 &

# R2: General Attention (10k)
nohup python train.py --model_type rnn --data_dir data/processed_10k --attention_type general --teacher_forcing_ratio 1.0 --name general_10k > log/rnn_general_10k_train.log 2>&1 &

# R3: Additive Attention (10k)
nohup python train.py --model_type rnn --data_dir data/processed_10k --attention_type additive --teacher_forcing_ratio 1.0 --name additive_10k > log/rnn_additive_10k_train.log 2>&1 &
```

#### 步骤 3: 训练（100k 数据集）

```bash
# R1: Dot-product Attention (100k)
nohup python train.py --model_type rnn --data_dir data/processed_100k --attention_type dot --teacher_forcing_ratio 1.0 --name dot_100k > log/rnn_dot_100k_train.log 2>&1 &

# R2: General Attention (100k)
nohup python train.py --model_type rnn --data_dir data/processed_100k --attention_type general --teacher_forcing_ratio 1.0 --name general_100k > log/rnn_general_100k_train.log 2>&1 &

# R3: Additive Attention (100k)
nohup python train.py --model_type rnn --data_dir data/processed_100k --attention_type additive --teacher_forcing_ratio 1.0 --name additive_100k > log/rnn_additive_100k_train.log 2>&1 &
```

#### 步骤 4: 评估（Greedy + Beam Search）

```bash
# === 10k 数据集评估 ===
# Dot - Greedy
nohup python inference.py --checkpoint checkpoints/rnn/best_dot_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 1 > log/rnn_dot_10k_greedy.log 2>&1 &
# Dot - Beam Search
nohup python inference.py --checkpoint checkpoints/rnn/best_dot_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/rnn_dot_10k_beam.log 2>&1 &

# General - Greedy
nohup python inference.py --checkpoint checkpoints/rnn/best_general_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 1 > log/rnn_general_10k_greedy.log 2>&1 &
# General - Beam Search
nohup python inference.py --checkpoint checkpoints/rnn/best_general_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/rnn_general_10k_beam.log 2>&1 &

# Additive - Greedy
nohup python inference.py --checkpoint checkpoints/rnn/best_additive_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 1 > log/rnn_additive_10k_greedy.log 2>&1 &
# Additive - Beam Search
nohup python inference.py --checkpoint checkpoints/rnn/best_additive_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/rnn_additive_10k_beam.log 2>&1 &

# === 100k 数据集评估 ===
# Dot - Greedy
nohup python inference.py --checkpoint checkpoints/rnn/best_dot_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 1 > log/rnn_dot_100k_greedy.log 2>&1 &
# Dot - Beam Search
nohup python inference.py --checkpoint checkpoints/rnn/best_dot_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/rnn_dot_100k_beam.log 2>&1 &

# General - Greedy
nohup python inference.py --checkpoint checkpoints/rnn/best_general_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 1 > log/rnn_general_100k_greedy.log 2>&1 &
# General - Beam Search
nohup python inference.py --checkpoint checkpoints/rnn/best_general_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/rnn_general_100k_beam.log 2>&1 &

# Additive - Greedy
nohup python inference.py --checkpoint checkpoints/rnn/best_additive_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 1 > log/rnn_additive_100k_greedy.log 2>&1 &
# Additive - Beam Search
nohup python inference.py --checkpoint checkpoints/rnn/best_additive_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/rnn_additive_100k_beam.log 2>&1 &
```

---

### 实验 R4: Training 策略对比 (Teacher Forcing vs Free Running)

> **说明**：Teacher Forcing 模型已在 R2 (General Attention) 训练完成，只需额外训练 Free Running 模型

#### 步骤 1: 训练 Free Running 模型

```bash
# 10k 数据集
nohup python train.py --model_type rnn --data_dir data/processed_10k --attention_type general --teacher_forcing_ratio 0.0 --name free_running_10k > log/rnn_free_running_10k_train.log 2>&1 &

# 100k 数据集
nohup python train.py --model_type rnn --data_dir data/processed_100k --attention_type general --teacher_forcing_ratio 0.0 --name free_running_100k > log/rnn_free_running_100k_train.log 2>&1 &
```

#### 步骤 2: 评估

```bash
# === 10k 数据集 ===
# Free Running - Greedy
nohup python inference.py --checkpoint checkpoints/rnn/best_free_running_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 1 > log/rnn_free_running_10k_greedy.log 2>&1 &
# Free Running - Beam Search
nohup python inference.py --checkpoint checkpoints/rnn/best_free_running_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/rnn_free_running_10k_beam.log 2>&1 &

# === 100k 数据集 ===
# Free Running - Greedy
nohup python inference.py --checkpoint checkpoints/rnn/best_free_running_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 1 > log/rnn_free_running_100k_greedy.log 2>&1 &
# Free Running - Beam Search
nohup python inference.py --checkpoint checkpoints/rnn/best_free_running_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/rnn_free_running_100k_beam.log 2>&1 &
```

---

## Transformer 对比实验

> **注意**：以下每个实验需分别在 10k 和 100k 数据集上运行
>
> **命令行参数**：`--pos_encoding`, `--use_rmsnorm`, `--d_model`, `--num_heads`, `--num_layers`, `--d_ff`, `--batch_size`, `--learning_rate`

### 实验 T1-T2: Position Encoding 对比

#### 步骤 1: 修改配置文件 `configs/config.yaml`

**10k 数据集配置：**
```yaml
transformer:
  d_model: 256
  num_heads: 4
  num_layers: 2              # 浅层网络
  d_ff: 1024
  dropout: 0.3               # 强正则化
  batch_size: 32
  epochs: 40
  learning_rate: 0.0005      # 略大的LR
  warmup_steps: 500          # 缩短热身
```

**100k 数据集配置：**
```yaml
transformer:
  d_model: 512
  num_heads: 8
  num_layers: 4              # 4层比6层在100k上更稳
  d_ff: 2048
  dropout: 0.1               # 标准正则化
  batch_size: 64
  epochs: 20
  learning_rate: 0.0001      # 标准LR
  warmup_steps: 4000         # 标准热身
```

> **注意**：`pos_encoding` 和 `use_rmsnorm` 通过命令行参数指定

#### 步骤 2: 训练

```bash
# === 10k 数据集 ===
# T1: Absolute Position Encoding (10k) - 作为 Baseline
nohup python train.py --model_type transformer --data_dir data/processed_10k --pos_encoding absolute --name absolute_10k > log/transformer_absolute_10k_train.log 2>&1 &

# T2: Relative Position Encoding (10k)
nohup python train.py --model_type transformer --data_dir data/processed_10k --pos_encoding relative --name relative_10k > log/transformer_relative_10k_train.log 2>&1 &

# === 100k 数据集 ===
# T1: Absolute Position Encoding (100k)
nohup python train.py --model_type transformer --data_dir data/processed_100k --pos_encoding absolute --name absolute_100k > log/transformer_absolute_100k_train.log 2>&1 &

# T2: Relative Position Encoding (100k)
nohup python train.py --model_type transformer --data_dir data/processed_100k --pos_encoding relative --name relative_100k > log/transformer_relative_100k_train.log 2>&1 &
```

#### 步骤 3: 评估

```bash
# 10k
nohup python inference.py --checkpoint checkpoints/transformer/best_absolute_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_absolute_10k_eval.log 2>&1 &
nohup python inference.py --checkpoint checkpoints/transformer/best_relative_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_relative_10k_eval.log 2>&1 &

# 100k
nohup python inference.py --checkpoint checkpoints/transformer/best_absolute_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_absolute_100k_eval.log 2>&1 &
nohup python inference.py --checkpoint checkpoints/transformer/best_relative_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_relative_100k_eval.log 2>&1 &
```

---

### 实验 T3: Normalization 对比

> LayerNorm 已在 T1 训练（best_absolute），只需额外训练 RMSNorm

#### 步骤 1: 训练 RMSNorm 模型

```bash
# 10k 数据集
nohup python train.py --model_type transformer --data_dir data/processed_10k --pos_encoding absolute --use_rmsnorm --name rmsnorm_10k > log/transformer_rmsnorm_10k_train.log 2>&1 &

# 100k 数据集
nohup python train.py --model_type transformer --data_dir data/processed_100k --pos_encoding absolute --use_rmsnorm --name rmsnorm_100k > log/transformer_rmsnorm_100k_train.log 2>&1 &
```

#### 步骤 2: 评估

```bash
# LayerNorm (复用 T1 的 best_absolute)
nohup python inference.py --checkpoint checkpoints/transformer/best_absolute_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_layernorm_10k_eval.log 2>&1 &
nohup python inference.py --checkpoint checkpoints/transformer/best_absolute_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_layernorm_100k_eval.log 2>&1 &

# RMSNorm
nohup python inference.py --checkpoint checkpoints/transformer/best_rmsnorm_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_rmsnorm_10k_eval.log 2>&1 &
nohup python inference.py --checkpoint checkpoints/transformer/best_rmsnorm_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_rmsnorm_100k_eval.log 2>&1 &
```

---

### 实验 T4-T5: Batch Size 对比

> 对比 batch_size = 16 和 64（32 已在 T1 作为 baseline 训练）

#### 步骤 1: 训练

```bash
# === batch_size = 16 ===
# 10k
nohup python train.py --model_type transformer --data_dir data/processed_10k --pos_encoding absolute --batch_size 16 --name bs16_10k > log/transformer_bs16_10k_train.log 2>&1 &
# 100k
nohup python train.py --model_type transformer --data_dir data/processed_100k --pos_encoding absolute --batch_size 16 --name bs16_100k > log/transformer_bs16_100k_train.log 2>&1 &

# === batch_size = 64 ===
# 10k
nohup python train.py --model_type transformer --data_dir data/processed_10k --pos_encoding absolute --batch_size 64 --name bs64_10k > log/transformer_bs64_10k_train.log 2>&1 &
# 100k
nohup python train.py --model_type transformer --data_dir data/processed_100k --pos_encoding absolute --batch_size 64 --name bs64_100k > log/transformer_bs64_100k_train.log 2>&1 &
```

#### 步骤 2: 评估

```bash
# 10k
nohup python inference.py --checkpoint checkpoints/transformer/best_bs16_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_bs16_10k_eval.log 2>&1 &
nohup python inference.py --checkpoint checkpoints/transformer/best_absolute_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_bs32_10k_eval.log 2>&1 &
nohup python inference.py --checkpoint checkpoints/transformer/best_bs64_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_bs64_10k_eval.log 2>&1 &

# 100k
nohup python inference.py --checkpoint checkpoints/transformer/best_bs16_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_bs16_100k_eval.log 2>&1 &
nohup python inference.py --checkpoint checkpoints/transformer/best_absolute_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_bs32_100k_eval.log 2>&1 &
nohup python inference.py --checkpoint checkpoints/transformer/best_bs64_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_bs64_100k_eval.log 2>&1 &
```

---

### 实验 T6-T7: Learning Rate 对比

> 对比 lr = 5e-4 和 1e-3（1e-4 已在 T1 作为 baseline 训练）

#### 步骤 1: 训练

```bash
# === lr = 5e-4 ===
# 10k
nohup python train.py --model_type transformer --data_dir data/processed_10k --pos_encoding absolute --learning_rate 0.0005 --name lr5e4_10k > log/transformer_lr5e4_10k_train.log 2>&1 &
# 100k
nohup python train.py --model_type transformer --data_dir data/processed_100k --pos_encoding absolute --learning_rate 0.0005 --name lr5e4_100k > log/transformer_lr5e4_100k_train.log 2>&1 &

# === lr = 1e-3 ===
# 10k
nohup python train.py --model_type transformer --data_dir data/processed_10k --pos_encoding absolute --learning_rate 0.001 --name lr1e3_10k > log/transformer_lr1e3_10k_train.log 2>&1 &
# 100k
nohup python train.py --model_type transformer --data_dir data/processed_100k --pos_encoding absolute --learning_rate 0.001 --name lr1e3_100k > log/transformer_lr1e3_100k_train.log 2>&1 &
```

#### 步骤 2: 评估

```bash
# 10k
nohup python inference.py --checkpoint checkpoints/transformer/best_absolute_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_lr1e4_10k_eval.log 2>&1 &
nohup python inference.py --checkpoint checkpoints/transformer/best_lr5e4_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_lr5e4_10k_eval.log 2>&1 &
nohup python inference.py --checkpoint checkpoints/transformer/best_lr1e3_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_lr1e3_10k_eval.log 2>&1 &

# 100k
nohup python inference.py --checkpoint checkpoints/transformer/best_absolute_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_lr1e4_100k_eval.log 2>&1 &
nohup python inference.py --checkpoint checkpoints/transformer/best_lr5e4_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_lr5e4_100k_eval.log 2>&1 &
nohup python inference.py --checkpoint checkpoints/transformer/best_lr1e3_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_lr1e3_100k_eval.log 2>&1 &
```

---

### 实验 T8-T9: Model Scale 对比（可选）

> 对比 Small 和 Large（Base 已在 T1 作为 baseline 训练）
>
> ⚠️ Large 模型需要较大 GPU 内存，如内存不足可跳过

#### 步骤 1: 训练

```bash
# === Small (d_model=256, num_heads=4, num_layers=2, d_ff=1024) ===
# 10k
nohup python train.py --model_type transformer --data_dir data/processed_10k --pos_encoding absolute --d_model 256 --num_heads 4 --num_layers 2 --d_ff 1024 --name small_10k > log/transformer_small_10k_train.log 2>&1 &
# 100k
nohup python train.py --model_type transformer --data_dir data/processed_100k --pos_encoding absolute --d_model 256 --num_heads 4 --num_layers 2 --d_ff 1024 --name small_100k > log/transformer_small_100k_train.log 2>&1 &

# === Large (d_model=1024, num_heads=16, num_layers=6, d_ff=4096) ===
# 10k
nohup python train.py --model_type transformer --data_dir data/processed_10k --pos_encoding absolute --d_model 1024 --num_heads 16 --num_layers 6 --d_ff 4096 --name large_10k > log/transformer_large_10k_train.log 2>&1 &
# 100k
nohup python train.py --model_type transformer --data_dir data/processed_100k --pos_encoding absolute --d_model 1024 --num_heads 16 --num_layers 6 --d_ff 4096 --name large_100k > log/transformer_large_100k_train.log 2>&1 &
```

#### 步骤 2: 评估

```bash
# 10k
nohup python inference.py --checkpoint checkpoints/transformer/best_small_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_small_10k_eval.log 2>&1 &
nohup python inference.py --checkpoint checkpoints/transformer/best_absolute_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_base_10k_eval.log 2>&1 &
nohup python inference.py --checkpoint checkpoints/transformer/best_large_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_large_10k_eval.log 2>&1 &

# 100k
nohup python inference.py --checkpoint checkpoints/transformer/best_small_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_small_100k_eval.log 2>&1 &
nohup python inference.py --checkpoint checkpoints/transformer/best_absolute_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_base_100k_eval.log 2>&1 &
nohup python inference.py --checkpoint checkpoints/transformer/best_large_100k.pt --data_dir data/processed_100k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/transformer_large_100k_eval.log 2>&1 &
```

---

## T5 微调实验

> **注意**：T5 使用独立的训练脚本 `train_t5.py`，直接读取原始 JSONL 文件

### 实验 P1: T5 Fine-tuning

#### 步骤 1: 修改配置文件

```yaml
t5:
  model_name: "t5-small"      # 可选: t5-small / t5-base / t5-large
  batch_size: 16
  epochs: 10
  learning_rate: 0.00003
```

#### 步骤 2: 训练

```bash
# 10k 数据集
nohup python train_t5.py --data 10k > log/t5_10k_train.log 2>&1 &

# 100k 数据集
nohup python train_t5.py --data 100k --epochs 5 --batch_size 16 > log/t5_100k_train.log 2>&1 &

# 指定模型大小 (可选)
nohup python train_t5.py --data 100k --model t5-base --batch_size 8 --name base_100k > log/t5_base_100k_train.log 2>&1 &
```

**train_t5.py 参数说明：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data` | 数据集大小 (10k/100k) | 必填 |
| `--model` | t5-small/t5-base/t5-large | t5-small |
| `--epochs` | 训练轮数 | 10 |
| `--batch_size` | 批次大小 | 32 |
| `--lr` | 学习率 | 3e-4 |
| `--max_len` | 最大序列长度 | 128 |
| `--name` | checkpoint后缀 | 同--data |

#### 步骤 3: 评估

```bash
# 10k (使用 --t5 参数)
nohup python inference.py --t5 --checkpoint checkpoints/t5/best_10k.pt --input data/raw/test.jsonl --evaluate --beam_width 4 > log/t5_10k_eval.log 2>&1 &

# 100k
nohup python inference.py --t5 --t5_model t5-base --checkpoint checkpoints/t5/best_base_100k.pt --input data/raw/test.jsonl --evaluate --beam_width 4 > log/t5_base_100k_eval.log 2>&1 &

nohup python inference.py --t5 --t5_model t5-large --checkpoint checkpoints/t5/best_large_100k.pt --input data/raw/test.jsonl --evaluate --beam_width 4 > log/t5_large_100k_eval.log 2>&1 &

nohup python inference.py --t5 --t5_model t5-3b --checkpoint checkpoints/t5/best_3b_100k.pt --input data/raw/test.jsonl --evaluate --beam_width 4 > log/t5_3b_100k_eval.log 2>&1 &
```

**inference.py T5 参数说明：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--t5` | 启用 T5 模型推理 | - |
| `--t5_model` | T5 预训练模型名称 | t5-small |

---

## 模型对比分析

### 分析维度

完成所有实验后，从以下维度进行对比分析：

| 维度 | 对比内容 | 指标 |
|------|----------|------|
| 模型架构 | RNN vs Transformer vs T5 | 参数量、结构复杂度 |
| 训练效率 | 训练时间、收敛速度 | 每epoch时间、loss曲线 |
| 翻译性能 | BLEU分数 | BLEU, BLEU-1/2/3/4 |
| 推理效率 | 解码速度 | 推理延迟 (ms/sentence) |
| 可扩展性 | 模型规模影响 | Small/Base/Large 对比 |

### 获取模型参数量

训练时会自动输出参数量，也可通过以下代码获取：

```python
from models import build_model
import yaml

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

model = build_model('transformer', config.get('transformer', {}))
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")
```

---

## 实验记录表格

### 指标获取方式

| 指标 | 获取来源 | 示例输出 |
|------|----------|----------|
| **BLEU / BLEU-1~4** | `inference.py --evaluate` 输出 | `BLEU: 12.34 (1:45.2 2:20.1 3:10.5 4:5.2)` |
| **参数量** | 训练开始时自动输出 | `Parameters: 12,345,678` |
| **训练时间** | 累加每个 epoch 的 Time | `Epoch X \| Time: 106.1s` |
| **收敛Epoch** | checkpoint 中的 epoch | 见下方命令 |
| **推理延迟** | 单独测量 | 见下方命令 |
| **GPU内存** | nvidia-smi | `nvidia-smi` |

#### 查看 checkpoint 信息（收敛Epoch、Val Loss）

```bash
python -c "
import torch
ckpt = torch.load('checkpoints/rnn/best_dot_10k.pt', map_location='cpu')
print('Epoch:', ckpt.get('epoch'))
print('Val Loss:', ckpt.get('val_loss'))
print('Config:', ckpt.get('config'))
"
```

#### 测量推理延迟

```bash
python -c "
import time
from inference import NMTInference

nmt = NMTInference('checkpoints/rnn/best_dot_10k.pt', 'data/processed_10k')
test_sentences = ['今天天气很好', '我喜欢学习', '这是一个测试']

# 预热
for s in test_sentences:
    nmt.translate(s, beam_width=5)

# 测量
start = time.time()
n_runs = 100
for _ in range(n_runs):
    for s in test_sentences:
        nmt.translate(s, beam_width=5)
elapsed = (time.time() - start) / (n_runs * len(test_sentences)) * 1000
print(f'推理延迟: {elapsed:.1f} ms/sentence')
"
```

#### 监控 GPU 内存

```bash
# 训练时在另一个终端运行
nvidia-smi --query-gpu=memory.used --format=csv -l 1
```

---

### RNN 实验记录 (10k 数据集)

| 实验ID | Attention | Training | Decoding | BLEU | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | 参数量 | 训练时间 | 推理延迟 |
|--------|-----------|----------|----------|------|--------|--------|--------|--------|--------|----------|----------|
| R1a-10k | dot | TF | Greedy | | | | | | | | |
| R1b-10k | dot | TF | Beam | | | | | | | | |
| R2a-10k | general | TF | Greedy | | | | | | | | |
| R2b-10k | general | TF | Beam | | | | | | | | |
| R3a-10k | additive | TF | Greedy | | | | | | | | |
| R3b-10k | additive | TF | Beam | | | | | | | | |
| R4a-10k | general | FR | Greedy | | | | | | | | |
| R4b-10k | general | FR | Beam | | | | | | | | |

### RNN 实验记录 (100k 数据集)

| 实验ID | Attention | Training | Decoding | BLEU | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | 参数量 | 训练时间 | 推理延迟 |
|--------|-----------|----------|----------|------|--------|--------|--------|--------|--------|----------|----------|
| R1a-100k | dot | TF | Greedy | | | | | | | | |
| R1b-100k | dot | TF | Beam | | | | | | | | |
| R2a-100k | general | TF | Greedy | | | | | | | | |
| R2b-100k | general | TF | Beam | | | | | | | | |
| R3a-100k | additive | TF | Greedy | | | | | | | | |
| R3b-100k | additive | TF | Beam | | | | | | | | |
| R4a-100k | general | FR | Greedy | | | | | | | | |
| R4b-100k | general | FR | Beam | | | | | | | | |

### Transformer Position Encoding 实验记录

| 实验ID | 数据集 | Position | BLEU | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | 参数量 | 训练时间 |
|--------|--------|----------|------|--------|--------|--------|--------|--------|----------|
| T1-10k | 10k | absolute | | | | | | | |
| T2-10k | 10k | relative | | | | | | | |
| T1-100k | 100k | absolute | | | | | | | |
| T2-100k | 100k | relative | | | | | | | |

### Transformer Normalization 实验记录

| 实验ID | 数据集 | Norm | BLEU | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | 参数量 | 训练时间 |
|--------|--------|------|------|--------|--------|--------|--------|--------|----------|
| T3-10k | 10k | LayerNorm | | | | | | | |
| T3-100k | 100k | LayerNorm | | | | | | | |
| T3-10k | 10k | RMSNorm | | | | | | | |
| T3-100k | 100k | RMSNorm | | | | | | | |

### Transformer Batch Size 实验记录

| 实验ID | 数据集 | Batch Size | BLEU | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | 训练时间 | 收敛Epoch |
|--------|--------|------------|------|--------|--------|--------|--------|----------|-----------|
| T4-10k | 10k | 16 | | | | | | | |
| T4-100k | 100k | 16 | | | | | | | |
| baseline | 10k | 32 | | | | | | | |
| baseline | 100k | 32 | | | | | | | |
| T5-10k | 10k | 64 | | | | | | | |
| T5-100k | 100k | 64 | | | | | | | |

### Transformer Learning Rate 实验记录

| 实验ID | 数据集 | Learning Rate | BLEU | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | 训练时间 | 收敛Epoch |
|--------|--------|---------------|------|--------|--------|--------|--------|----------|-----------|
| baseline | 10k | 1e-4 | | | | | | | |
| baseline | 100k | 1e-4 | | | | | | | |
| T6-10k | 10k | 5e-4 | | | | | | | |
| T6-100k | 100k | 5e-4 | | | | | | | |
| T7-10k | 10k | 1e-3 | | | | | | | |
| T7-100k | 100k | 1e-3 | | | | | | | |

### Transformer Model Scale 实验记录（可选）

| 实验ID | 数据集 | Scale | d_model | layers | d_ff | BLEU | 参数量 | 训练时间 | GPU内存 |
|--------|--------|-------|---------|--------|------|------|--------|----------|---------|
| T8-10k | 10k | small | 256 | 2 | 1024 | | | | |
| T8-100k | 100k | small | 256 | 2 | 1024 | | | | |
| baseline | 10k | base | 512 | 6 | 2048 | | | | |
| baseline | 100k | base | 512 | 6 | 2048 | | | | |
| T9-10k | 10k | large | 1024 | 6 | 4096 | | | | |
| T9-100k | 100k | large | 1024 | 6 | 4096 | | | | |

### T5 Fine-tuning 实验记录

| 实验ID | 数据集 | 模型 | BLEU | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | 参数量 | 训练时间 |
|--------|--------|------|------|--------|--------|--------|--------|--------|----------|
| P1-10k | 10k | t5-small | | | | | | | |
| P1-100k | 100k | t5-small | | | | | | | |

### 最终模型对比汇总 (10k 数据集)

| 模型类型 | 最佳配置 | BLEU | 参数量 | 训练时间 | 推理延迟 | 优势 | 劣势 |
|----------|----------|------|--------|----------|----------|------|------|
| RNN | | | | | | | |
| Transformer | | | | | | | |
| T5 | | | | | | | |

### 最终模型对比汇总 (100k 数据集)

| 模型类型 | 最佳配置 | BLEU | 参数量 | 训练时间 | 推理延迟 | 优势 | 劣势 |
|----------|----------|------|--------|----------|----------|------|------|
| RNN | | | | | | | |
| Transformer | | | | | | | |
| T5 | | | | | | | |

### 数据规模影响对比

| 模型类型 | 10k BLEU | 100k BLEU | 提升幅度 | 训练时间增加 |
|----------|----------|-----------|----------|--------------|
| RNN (best) | | | | |
| Transformer (best) | | | | |
| T5 | | | | |

---

## 快速参考

### 配置文件模板

完整配置文件 `configs/config.yaml`:

```yaml
# 通用设置
max_length: 128
vocab_size: 5000
device: "cuda"
seed: 42
checkpoint_dir: "checkpoints"
save_every: 5
use_amp: false

# RNN配置
rnn:
  embed_size: 256
  hidden_size: 512
  num_layers: 2
  rnn_type: "gru"              # gru / lstm
  attention_type: "general"    # dot / general / additive
  dropout: 0.3
  batch_size: 64
  epochs: 30
  learning_rate: 0.001
  teacher_forcing_ratio: 1.0   # 1.0 = TF, 0.0 = FR
  tf_decay: false

# Transformer配置
transformer:
  d_model: 512
  num_heads: 8
  num_layers: 6
  d_ff: 2048
  dropout: 0.1
  pos_encoding: "absolute"     # absolute / relative
  use_rmsnorm: false           # false = LayerNorm, true = RMSNorm
  batch_size: 32
  epochs: 20
  learning_rate: 0.0001
  warmup_steps: 4000

# T5配置
t5:
  model_name: "t5-small"
  batch_size: 16
  epochs: 10
  learning_rate: 0.00003
```

### 常用命令速查

```bash
# === 数据准备 ===
python data/scripts/prepare_data2.py --data 10k
python data/scripts/prepare_data2.py --data 100k --vocab_size 10000 --min_freq 2 --max_length 100

# === 训练 (10k) ===
nohup python train.py --model_type rnn --data_dir data/processed_10k --attention_type general --teacher_forcing_ratio 1.0 --name general_10k > log/rnn_10k_train.log 2>&1 &
nohup python train.py --model_type transformer --data_dir data/processed_10k --pos_encoding absolute --name absolute_10k > log/transformer_10k_train.log 2>&1 &
nohup python train_t5.py --data 10k > log/t5_10k_train.log 2>&1 &

# === 训练 (100k) ===
nohup python train.py --model_type rnn --data_dir data/processed_100k --attention_type general --teacher_forcing_ratio 1.0 --name general_100k > log/rnn_100k_train.log 2>&1 &
nohup python train.py --model_type transformer --data_dir data/processed_100k --pos_encoding absolute --name absolute_100k > log/transformer_100k_train.log 2>&1 &
nohup python train_t5.py --data 100k --epochs 5 > log/t5_100k_train.log 2>&1 &

# === 评估 (Greedy) ===
nohup python inference.py --checkpoint checkpoints/rnn/best_general_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 1 > log/rnn_general_10k_greedy.log 2>&1 &

# === 评估 (Beam Search) ===
nohup python inference.py --checkpoint checkpoints/rnn/best_general_10k.pt --data_dir data/processed_10k --input data/raw/test.jsonl --evaluate --beam_width 5 > log/rnn_general_10k_beam.log 2>&1 &

# === T5 评估 (使用 --t5 参数) ===
nohup python inference.py --t5 --checkpoint checkpoints/t5/best_10k.pt --input data/raw/test.jsonl --evaluate --beam_width 4 > log/t5_10k_eval.log 2>&1 &

# === 交互翻译 ===
python inference.py --checkpoint checkpoints/transformer/best_absolute_100k.pt --data_dir data/processed_100k --interactive

# === T5 交互翻译 ===
python inference.py --t5 --checkpoint checkpoints/t5/best_10k.pt --interactive

# === 单句翻译 ===
python inference.py --checkpoint checkpoints/transformer/best_absolute_100k.pt --data_dir data/processed_100k --text "今天天气真热，气温很高"

# === T5 单句翻译 ===
python inference.py --t5 --checkpoint checkpoints/t5/best_10k.pt --text "今天天气真热，气温很高"

# === 查看后台任务 ===
jobs -l

# === 实时查看日志 ===
tail -f log/xxx.log
```

### 实验执行顺序建议

1. **第一阶段：10k 数据集快速验证**
   - 运行所有实验，验证代码和配置正确
   - 预计时间：~5小时

2. **第二阶段：100k 数据集正式实验**
   - 运行所有实验，获取正式结果
   - 预计时间：~40小时

3. **第三阶段：结果分析**
   - 填写实验记录表格
   - 对比10k和100k结果
   - 撰写分析报告

---

## 参考文献

- Attention Is All You Need (Vaswani et al., 2017)
- Effective Approaches to Attention-based NMT (Luong et al., 2015)
- Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2015)
- Exploring the Limits of Transfer Learning with T5 (Raffel et al., 2020)
