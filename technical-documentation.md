# Technical Documentation: Haiku RNN

## Overview

Haiku RNN is a character-level recurrent neural network designed to generate Japanese haiku poetry. Based on Andrej Karpathy's char-rnn and adapted for TensorFlow by Sherjil Ozair, this implementation adds UTF-16 encoding support to properly handle Japanese characters.

**Author:** Henry Wolf (aenrichus@gmail.com)
**License:** MIT License (Copyright 2017)

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Architecture](#architecture)
3. [Data Format and Preprocessing](#data-format-and-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Text Generation](#text-generation)
7. [Dependencies](#dependencies)
8. [Pre-trained Models](#pre-trained-models)
9. [Next Steps for Modernization](#next-steps-for-modernization)

---

## Project Structure

```
haiku_rnn/
├── README.md                      # Project overview
├── LICENSE                        # MIT License
├── technical-documentation.md     # This file
├── model.py                       # RNN model architecture (99 lines)
├── train.py                       # Training script (111 lines)
├── sample.py                      # Text generation script (44 lines)
├── utils.py                       # Data loading utilities (72 lines)
├── data/
│   ├── basho-utf16/
│   │   ├── input.txt             # Matsuo Basho haiku corpus
│   │   ├── vocab.pkl             # Character vocabulary
│   │   └── data.npy              # Preprocessed tensor
│   └── issa-utf16/
│       ├── input.txt             # Kobayashi Issa haiku corpus
│       ├── vocab.pkl             # Character vocabulary
│       └── data.npy              # Preprocessed tensor
├── saveBasho/                     # Pre-trained Basho model
│   ├── config.pkl                # Model hyperparameters
│   ├── chars_vocab.pkl           # Character mappings
│   └── model.ckpt-*              # TensorFlow checkpoints
├── saveIssa/                      # Pre-trained Issa model
│   ├── config.pkl                # Model hyperparameters
│   ├── chars_vocab.pkl           # Character mappings
│   └── model.ckpt-*              # TensorFlow checkpoints
└── save/                          # Custom model directory
```

---

## Architecture

### High-Level Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ UTF-16 Text │───▶│ TextLoader   │───▶│ Char Tensor │───▶│ RNN Model    │
│ (input.txt) │    │ (utils.py)   │    │ (data.npy)  │    │ (model.py)   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                          │                                       │
                          ▼                                       ▼
                   ┌──────────────┐                        ┌──────────────┐
                   │ Vocabulary   │                        │ Trained      │
                   │ (vocab.pkl)  │                        │ Weights      │
                   └──────────────┘                        └──────────────┘
```

### Component Responsibilities

| File | Responsibility |
|------|----------------|
| `utils.py` | Data loading, preprocessing, batching |
| `model.py` | RNN architecture definition, sampling logic |
| `train.py` | Training loop, checkpointing, hyperparameters |
| `sample.py` | Model loading, text generation interface |

---

## Data Format and Preprocessing

### Input Format

- **Encoding:** UTF-16 (big-endian)
- **Structure:** Haiku separated by double newlines (`\n\n`)
- **Format:** Original Japanese text with line breaks

### Sample Data (Basho)

```
於春々大哉春と云々

青くてもあるべきものを唐辛子

青ざしや草餅の穂に出でつらん
```

### Dataset Statistics

| Dataset | Haiku Count | File Size | Unique Characters |
|---------|-------------|-----------|-------------------|
| Basho   | ~1,067      | 31 KB     | 1,263            |
| Issa    | ~1,206      | 1.3 MB    | 2,333            |

### Preprocessing Pipeline (`TextLoader` class)

1. **Read:** Load UTF-16 encoded text file using `codecs.open()`
2. **Vocabulary Build:** Count character frequencies with `collections.Counter`
3. **Mapping Creation:** Create bidirectional char↔index mappings
4. **Tensorization:** Convert text to NumPy array of indices
5. **Caching:** Save `vocab.pkl` and `data.npy` for reuse

```python
# Core preprocessing logic
counter = collections.Counter(data)
vocab = sorted(counter.items(), key=lambda x: -x[1])
self.chars, _ = zip(*vocab)
self.vocab = dict(zip(self.chars, range(len(self.chars))))
self.tensor = np.array(list(map(self.vocab.get, data)))
```

### Batching Strategy

Data is reshaped into sequential batches for training:

```python
# Batch creation
self.tensor = self.tensor[:num_batches * batch_size * seq_length]
xdata = self.tensor
ydata = np.copy(self.tensor)
ydata[:-1] = xdata[1:]  # Target is next character
ydata[-1] = xdata[0]
self.x_batches = np.split(xdata.reshape(batch_size, -1), num_batches, 1)
self.y_batches = np.split(ydata.reshape(batch_size, -1), num_batches, 1)
```

---

## Model Architecture

### Neural Network Structure

```
Input (char index) ─────┐
                        ▼
               ┌────────────────┐
               │   Embedding    │  (vocab_size → rnn_size)
               │     Layer      │
               └───────┬────────┘
                       ▼
               ┌────────────────┐
               │   LSTM Layer   │  (rnn_size=128)
               │      #1        │
               └───────┬────────┘
                       ▼
               ┌────────────────┐
               │   LSTM Layer   │  (rnn_size=128)
               │      #2        │
               └───────┬────────┘
                       ▼
               ┌────────────────┐
               │    Softmax     │  (rnn_size → vocab_size)
               │     Layer      │
               └───────┬────────┘
                       ▼
             Character Probabilities
```

### Supported Cell Types

| Cell Type | TensorFlow Class | Description |
|-----------|------------------|-------------|
| `rnn`     | `BasicRNNCell`   | Simple recurrent unit |
| `gru`     | `GRUCell`        | Gated Recurrent Unit |
| `lstm`    | `BasicLSTMCell`  | Long Short-Term Memory (default) |

### Model Hyperparameters

| Parameter | Basho Model | Issa Model | Description |
|-----------|-------------|------------|-------------|
| `model`   | lstm        | lstm       | RNN cell type |
| `rnn_size`| 128         | 128        | Hidden state dimension |
| `num_layers`| 2         | 2          | Stacked RNN layers |
| `batch_size`| 20        | 50         | Training batch size |
| `seq_length`| 20        | 50         | Sequence length per batch |
| `vocab_size`| 1,263     | 2,333      | Character vocabulary size |

### Key Implementation Details

**Embedding Layer:**
```python
embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
inputs = tf.split(tf.nn.embedding_lookup(embedding, self.input_data),
                  args.seq_length, 1)
```

**Multi-Layer RNN:**
```python
cell = rnn_cell.BasicLSTMCell(args.rnn_size)
self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)
```

**Output Projection:**
```python
softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
self.probs = tf.nn.softmax(tf.matmul(output, softmax_w) + softmax_b)
```

---

## Training Process

### Training Loop

```python
for e in range(args.num_epochs):
    # Learning rate decay
    sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))

    for b in range(data_loader.num_batches):
        x, y = data_loader.next_batch()
        feed = {model.input_data: x, model.targets: y,
                model.initial_state: state}
        train_loss, state, _ = sess.run(
            [model.cost, model.final_state, model.train_op], feed)
```

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 50 | Training iterations over dataset |
| `learning_rate` | 0.002 | Initial Adam learning rate |
| `decay_rate` | 0.97 | LR decay per epoch |
| `grad_clip` | 5.0 | Gradient clipping threshold |
| `save_every` | 1000 | Checkpoint frequency (batches) |

### Loss Function

Uses TensorFlow's `sequence_loss_by_example` with cross-entropy:

```python
loss = seq2seq.sequence_loss_by_example(
    [self.logits],
    [tf.reshape(self.targets, [-1])],
    [tf.ones([args.batch_size * args.seq_length])],
    args.vocab_size)
self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
```

### Optimizer

Adam optimizer with gradient clipping:

```python
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
optimizer = tf.train.AdamOptimizer(self.lr)
self.train_op = optimizer.apply_gradients(zip(grads, tvars))
```

### Training Command

```bash
python train.py \
    --data_dir data/issa-utf16 \
    --save_dir saveIssa \
    --model lstm \
    --rnn_size 128 \
    --num_layers 2 \
    --batch_size 50 \
    --seq_length 50 \
    --num_epochs 50
```

---

## Text Generation

### Sampling Strategies

The model supports three sampling modes via the `sample` parameter:

| Mode | Name | Description |
|------|------|-------------|
| 0 | Greedy | Always select highest probability character |
| 1 | Weighted | Sample proportionally to probability distribution |
| 2 | Conditional | Greedy except sample at word boundaries |

### Sampling Implementation

```python
def sample(self, sess, chars, vocab, num=200, prime=' ', sampling_type=1):
    state = sess.run(self.cell.zero_state(1, tf.float32))

    # Prime the network
    for char in prime[:-1]:
        x = np.zeros((1, 1))
        x[0, 0] = vocab[char]
        feed = {self.input_data: x, self.initial_state: state}
        [state] = sess.run([self.final_state], feed)

    # Generate characters
    ret = prime
    char = prime[-1]
    for _ in range(num):
        x = np.zeros((1, 1))
        x[0, 0] = vocab[char]
        feed = {self.input_data: x, self.initial_state: state}
        [probs, state] = sess.run([self.probs, self.final_state], feed)

        # Sample based on strategy
        if sampling_type == 0:
            sample = np.argmax(probs[0])
        elif sampling_type == 1:
            sample = weighted_pick(probs[0])
        # ...

        char = chars[sample]
        ret += char
    return ret
```

### Weighted Sampling Function

```python
def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return int(np.searchsorted(t, np.random.rand(1) * s))
```

### Generation Command

```bash
python sample.py \
    --save_dir saveIssa \
    -n 2000 \
    --prime " " \
    --sample 1
```

---

## Dependencies

### Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| TensorFlow | 1.x | Deep learning framework |
| NumPy | Any | Array operations |
| Six | Any | Python 2/3 compatibility |

### TensorFlow API Components Used

- `tensorflow.python.ops.rnn_cell` - RNN cell implementations
- `tensorflow.python.ops.seq2seq` - Sequence loss functions
- `tf.train.Saver` - Model checkpointing
- `tf.train.AdamOptimizer` - Optimization

### Standard Library

- `codecs` - UTF-16 file handling
- `collections.Counter` - Character counting
- `pickle` (via `six.moves.cPickle`) - Serialization
- `argparse` - CLI argument parsing

---

## Pre-trained Models

### Basho Model (`saveBasho/`)

- **Training Progress:** 3,899 steps (100 epochs completed)
- **Vocabulary:** 1,263 unique characters
- **Checkpoints:** 0, 1000, 2000, 3000, 3899

### Issa Model (`saveIssa/`)

- **Training Progress:** 13,049 steps (50 epochs completed)
- **Vocabulary:** 2,333 unique characters
- **Checkpoints:** 10000, 11000, 12000, 13000, 13049

---

## Next Steps for Modernization

This section outlines a roadmap for updating the haiku_rnn project to use modern deep learning practices and frameworks.

### Phase 1: PyTorch Port (Priority: High)

The current TensorFlow 1.x implementation uses deprecated APIs. Porting to PyTorch provides better maintainability and modern tooling.

#### 1.1 Create PyTorch Model Architecture

```python
# Proposed PyTorch model structure
import torch
import torch.nn as nn

class HaikuRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=128,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embed = self.embedding(x)
        output, hidden = self.lstm(embed, hidden)
        logits = self.fc(output)
        return logits, hidden
```

#### 1.2 Migration Tasks

- [ ] Convert `model.py` to PyTorch `nn.Module`
- [ ] Replace `utils.py` with PyTorch `Dataset` and `DataLoader`
- [ ] Update `train.py` to use PyTorch training loop
- [ ] Update `sample.py` for PyTorch inference
- [ ] Add `requirements.txt` with PyTorch dependencies
- [ ] Implement checkpoint conversion utility for existing models

#### 1.3 Preserve Compatibility

- Maintain UTF-16 encoding support
- Keep identical sampling strategies
- Ensure reproducible results with seeded random states

### Phase 2: Modern Architecture Improvements (Priority: Medium)

#### 2.1 Transformer-Based Architecture

Consider migrating to a transformer architecture for improved generation quality:

```python
class HaikuTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
```

#### 2.2 Architecture Options

| Architecture | Pros | Cons |
|--------------|------|------|
| LSTM (current) | Simple, low memory | Limited context |
| GRU | Faster than LSTM | Similar limitations |
| Transformer | Long-range deps | Higher compute |
| GPT-style | State-of-art quality | Requires more data |

### Phase 3: Training Improvements (Priority: Medium)

#### 3.1 Modern Training Techniques

- [ ] Implement mixed-precision training (FP16)
- [ ] Add learning rate schedulers (cosine annealing, warmup)
- [ ] Implement gradient accumulation for larger effective batch sizes
- [ ] Add proper train/validation/test splits
- [ ] Implement early stopping with patience
- [ ] Add TensorBoard/Weights & Biases logging

#### 3.2 Regularization

- [ ] Add dropout between LSTM layers
- [ ] Implement weight decay
- [ ] Add layer normalization

#### 3.3 Training Script Improvements

```python
# Proposed modern training loop structure
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    for batch in dataloader:
        with autocast():
            logits, _ = model(batch.input)
            loss = criterion(logits.view(-1, vocab_size), batch.target.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()
```

### Phase 4: Data Pipeline Modernization (Priority: Medium)

#### 4.1 PyTorch Dataset Implementation

```python
class HaikuDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_length=50):
        self.seq_length = seq_length
        self.data, self.vocab, self.chars = self.load_data(data_path)

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x), torch.tensor(y)
```

#### 4.2 Data Improvements

- [ ] Implement proper train/val/test splits
- [ ] Add data augmentation (character-level noise)
- [ ] Support additional encodings beyond UTF-16
- [ ] Add streaming data loading for larger corpora

### Phase 5: Generation Improvements (Priority: Medium)

#### 5.1 Advanced Sampling Methods

- [ ] Implement temperature-based sampling
- [ ] Add top-k sampling
- [ ] Add nucleus (top-p) sampling
- [ ] Implement beam search for deterministic outputs

```python
def sample_with_temperature(logits, temperature=1.0):
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1)

def top_k_sampling(logits, k=50):
    values, indices = torch.topk(logits, k)
    probs = F.softmax(values, dim=-1)
    sampled = torch.multinomial(probs, 1)
    return indices.gather(-1, sampled)

def nucleus_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    probs = F.softmax(sorted_logits, dim=-1)
    sampled = torch.multinomial(probs, 1)
    return sorted_indices.gather(-1, sampled)
```

#### 5.2 Haiku-Specific Generation

- [ ] Add syllable counting for proper 5-7-5 structure
- [ ] Implement constrained generation for valid haiku format
- [ ] Add kireji (cutting word) awareness

### Phase 6: Expanded Dataset (Priority: High)

#### 6.1 Data Collection

- [ ] Aggregate additional haiku corpora (public domain)
- [ ] Clean and normalize existing Issa dataset
- [ ] Add romaji/hiragana/katakana variants
- [ ] Consider bilingual (Japanese/English) training

#### 6.2 Data Quality

- [ ] Remove web metadata artifacts from training data
- [ ] Implement deduplication
- [ ] Add proper haiku format validation

### Phase 7: Deployment and Usability (Priority: Low)

#### 7.1 Model Serving

- [ ] Export to ONNX format for cross-platform deployment
- [ ] Create REST API with FastAPI/Flask
- [ ] Add Docker containerization
- [ ] Implement model quantization for edge deployment

#### 7.2 User Interface

- [ ] Create CLI with rich formatting
- [ ] Add web-based generation interface
- [ ] Implement batch generation mode

### Phase 8: Testing and Documentation (Priority: Medium)

#### 8.1 Testing

- [ ] Add unit tests for model components
- [ ] Add integration tests for training pipeline
- [ ] Add generation quality benchmarks
- [ ] Implement perplexity evaluation metrics

#### 8.2 Documentation

- [ ] Add docstrings to all functions
- [ ] Create API documentation
- [ ] Add example notebooks
- [ ] Document model performance metrics

---

## Implementation Priority Summary

| Phase | Priority | Effort | Impact |
|-------|----------|--------|--------|
| PyTorch Port | High | Medium | High |
| Expanded Dataset | High | High | High |
| Training Improvements | Medium | Medium | Medium |
| Modern Architecture | Medium | High | High |
| Generation Improvements | Medium | Low | Medium |
| Data Pipeline | Medium | Medium | Medium |
| Testing/Documentation | Medium | Medium | Medium |
| Deployment | Low | Medium | Low |

---

## Quick Start After Modernization

```bash
# Install dependencies
pip install torch numpy tqdm

# Train a new model
python train.py --data_dir data/issa-utf16 --epochs 50 --device cuda

# Generate haiku
python sample.py --checkpoint models/best.pt --num_chars 500 --temperature 0.8
```

---

## References

- [Karpathy's char-rnn](https://github.com/karpathy/char-rnn)
- [Sherjil Ozair's TensorFlow char-rnn](https://github.com/sherjilozair/char-rnn-tensorflow)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
