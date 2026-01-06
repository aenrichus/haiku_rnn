#!/usr/bin/env python3
"""
PyTorch implementation of Haiku RNN - A character-level RNN for generating Japanese haiku.

This is a modern PyTorch port of the original TensorFlow implementation,
with support for Apple Silicon (MPS), CUDA, and CPU backends.

Usage:
    Training:
        python model_pytorch.py train --data_dir data/issa-utf16 --save_dir save_pytorch

    Sampling:
        python model_pytorch.py sample --save_dir save_pytorch -n 500

Author: Ported to PyTorch from original by Henry Wolf
"""

import argparse
import codecs
import collections
import os
import pickle
import time
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Device Detection
# =============================================================================

def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Get the best available device for training/inference.

    Priority: MPS (Apple Silicon) > CUDA > CPU

    Args:
        preferred: Optional preferred device ('mps', 'cuda', 'cpu')

    Returns:
        torch.device: The selected device
    """
    if preferred:
        if preferred == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        elif preferred == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif preferred == 'cpu':
            return torch.device('cpu')
        else:
            print(f"Warning: Preferred device '{preferred}' not available, auto-selecting...")

    # Auto-detect best device
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) backend")
        return torch.device('mps')
    elif torch.cuda.is_available():
        print(f"Using CUDA backend ({torch.cuda.get_device_name(0)})")
        return torch.device('cuda')
    else:
        print("Using CPU backend")
        return torch.device('cpu')


# =============================================================================
# Dataset
# =============================================================================

class HaikuDataset(Dataset):
    """
    PyTorch Dataset for character-level haiku text data.

    Handles UTF-16 encoded text files and provides character-to-index mappings.
    """

    def __init__(
        self,
        data_dir: str,
        seq_length: int = 50,
        encoding: str = 'utf-16'
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing input.txt
            seq_length: Length of each training sequence
            encoding: Text file encoding (default: utf-16)
        """
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("Reading and preprocessing text file...")
            self._preprocess(input_file, vocab_file, tensor_file)
        else:
            print("Loading preprocessed files...")
            self._load_preprocessed(vocab_file, tensor_file)

    def _preprocess(self, input_file: str, vocab_file: str, tensor_file: str):
        """Read raw text and create vocabulary and tensor."""
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()

        # Build vocabulary from character frequencies
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.chars = list(self.chars)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

        # Save vocabulary
        with open(vocab_file, 'wb') as f:
            pickle.dump(self.chars, f)

        # Convert text to tensor of indices
        self.tensor = np.array([self.vocab[c] for c in data], dtype=np.int64)
        np.save(tensor_file, self.tensor)

        print(f"Vocabulary size: {self.vocab_size} characters")
        print(f"Total characters: {len(self.tensor)}")

    def _load_preprocessed(self, vocab_file: str, tensor_file: str):
        """Load preprocessed vocabulary and tensor."""
        with open(vocab_file, 'rb') as f:
            self.chars = pickle.load(f)

        # Handle both list and tuple formats
        if isinstance(self.chars, tuple):
            self.chars = list(self.chars)

        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file).astype(np.int64)

        print(f"Vocabulary size: {self.vocab_size} characters")
        print(f"Total characters: {len(self.tensor)}")

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return max(0, len(self.tensor) - self.seq_length)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training example.

        Args:
            idx: Index of the sequence

        Returns:
            Tuple of (input_sequence, target_sequence)
            Target is input shifted by one character
        """
        x = torch.tensor(self.tensor[idx:idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.tensor[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return x, y

    def get_vocab_info(self) -> Tuple[List[str], Dict[str, int], int]:
        """Return vocabulary information."""
        return self.chars, self.vocab, self.vocab_size


# =============================================================================
# Model
# =============================================================================

class HaikuRNN(nn.Module):
    """
    Character-level RNN for haiku generation.

    Supports RNN, GRU, and LSTM cell types with configurable layers and hidden size.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        model_type: str = 'lstm',
        dropout: float = 0.0
    ):
        """
        Initialize the model.

        Args:
            vocab_size: Size of the character vocabulary
            embed_size: Dimension of character embeddings
            hidden_size: Dimension of RNN hidden state
            num_layers: Number of stacked RNN layers
            model_type: Type of RNN cell ('rnn', 'gru', 'lstm')
            dropout: Dropout probability between RNN layers
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type.lower()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # RNN layer
        rnn_dropout = dropout if num_layers > 1 else 0.0

        if self.model_type == 'rnn':
            self.rnn = nn.RNN(
                embed_size, hidden_size, num_layers,
                batch_first=True, dropout=rnn_dropout
            )
        elif self.model_type == 'gru':
            self.rnn = nn.GRU(
                embed_size, hidden_size, num_layers,
                batch_first=True, dropout=rnn_dropout
            )
        elif self.model_type == 'lstm':
            self.rnn = nn.LSTM(
                embed_size, hidden_size, num_layers,
                batch_first=True, dropout=rnn_dropout
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'rnn', 'gru', or 'lstm'")

        # Output projection layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of character indices (batch_size, seq_length)
            hidden: Optional initial hidden state

        Returns:
            Tuple of (logits, hidden_state)
            logits shape: (batch_size, seq_length, vocab_size)
        """
        # Embed characters
        embed = self.embedding(x)  # (batch, seq, embed_size)

        # Pass through RNN
        output, hidden = self.rnn(embed, hidden)  # (batch, seq, hidden_size)

        # Project to vocabulary
        logits = self.fc(output)  # (batch, seq, vocab_size)

        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """
        Initialize hidden state with zeros.

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            Initial hidden state (format depends on RNN type)
        """
        if self.model_type == 'lstm':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return (h0, c0)
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return (h0,)

    def detach_hidden(self, hidden: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Detach hidden state from computation graph (for TBPTT)."""
        if self.model_type == 'lstm':
            return (hidden[0].detach(), hidden[1].detach())
        else:
            return (hidden[0].detach(),)


# =============================================================================
# Sampling / Generation
# =============================================================================

def weighted_pick(weights: np.ndarray) -> int:
    """
    Sample an index from a probability distribution.

    Args:
        weights: Probability distribution (should sum to 1)

    Returns:
        Sampled index
    """
    t = np.cumsum(weights)
    s = np.sum(weights)
    return int(np.searchsorted(t, np.random.rand() * s))


def sample(
    model: HaikuRNN,
    chars: List[str],
    vocab: Dict[str, int],
    device: torch.device,
    num_chars: int = 500,
    prime: str = ' ',
    temperature: float = 1.0,
    sampling_type: int = 1
) -> str:
    """
    Generate text using the trained model.

    Args:
        model: Trained HaikuRNN model
        chars: List of characters (index to char mapping)
        vocab: Dictionary mapping characters to indices
        device: Device to run inference on
        num_chars: Number of characters to generate
        prime: Seed text to start generation
        temperature: Sampling temperature (higher = more random)
        sampling_type: 0=greedy, 1=weighted, 2=sample on spaces only

    Returns:
        Generated text string
    """
    model.eval()

    with torch.no_grad():
        # Initialize hidden state
        hidden = model.init_hidden(1, device)
        if model.model_type != 'lstm':
            hidden = hidden[0]  # Unwrap for RNN/GRU

        # Prime the network with seed text
        for char in prime[:-1]:
            if char not in vocab:
                print(f"Warning: Character '{char}' not in vocabulary, skipping")
                continue
            x = torch.tensor([[vocab[char]]], dtype=torch.long, device=device)
            _, hidden = model(x, hidden)

        # Start generating
        result = prime
        char = prime[-1] if prime else ' '

        for _ in range(num_chars):
            if char not in vocab:
                char = ' '  # Fallback to space if character unknown

            x = torch.tensor([[vocab[char]]], dtype=torch.long, device=device)
            logits, hidden = model(x, hidden)

            # Apply temperature
            logits = logits[0, 0] / temperature
            probs = F.softmax(logits, dim=0).cpu().numpy()

            # Sample based on strategy
            if sampling_type == 0:
                # Greedy: always pick most likely
                sample_idx = np.argmax(probs)
            elif sampling_type == 2:
                # Conditional: sample only at word boundaries
                if char == ' ':
                    sample_idx = weighted_pick(probs)
                else:
                    sample_idx = np.argmax(probs)
            else:
                # Weighted random sampling (default)
                sample_idx = weighted_pick(probs)

            char = chars[sample_idx]
            result += char

        return result


def top_k_sampling(logits: torch.Tensor, k: int = 50) -> int:
    """Top-k sampling: sample from the k most likely tokens."""
    values, indices = torch.topk(logits, k)
    probs = F.softmax(values, dim=-1)
    sampled = torch.multinomial(probs, 1)
    return indices[sampled].item()


def nucleus_sampling(logits: torch.Tensor, p: float = 0.9) -> int:
    """Nucleus (top-p) sampling: sample from tokens comprising top-p probability mass."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Keep at least one token
    sorted_indices_to_remove[0] = False

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    probs = F.softmax(sorted_logits, dim=-1)
    sampled = torch.multinomial(probs, 1)
    return sorted_indices[sampled].item()


def sample_advanced(
    model: HaikuRNN,
    chars: List[str],
    vocab: Dict[str, int],
    device: torch.device,
    num_chars: int = 500,
    prime: str = ' ',
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> str:
    """
    Generate text with advanced sampling methods.

    Args:
        model: Trained HaikuRNN model
        chars: List of characters (index to char mapping)
        vocab: Dictionary mapping characters to indices
        device: Device to run inference on
        num_chars: Number of characters to generate
        prime: Seed text to start generation
        temperature: Sampling temperature
        top_k: If set, use top-k sampling
        top_p: If set, use nucleus sampling

    Returns:
        Generated text string
    """
    model.eval()

    with torch.no_grad():
        hidden = model.init_hidden(1, device)
        if model.model_type != 'lstm':
            hidden = hidden[0]

        # Prime the network
        for char in prime[:-1]:
            if char not in vocab:
                continue
            x = torch.tensor([[vocab[char]]], dtype=torch.long, device=device)
            _, hidden = model(x, hidden)

        result = prime
        char = prime[-1] if prime else ' '

        for _ in range(num_chars):
            if char not in vocab:
                char = ' '

            x = torch.tensor([[vocab[char]]], dtype=torch.long, device=device)
            logits, hidden = model(x, hidden)
            logits = logits[0, 0] / temperature

            # Apply sampling strategy
            if top_k is not None:
                sample_idx = top_k_sampling(logits, k=top_k)
            elif top_p is not None:
                sample_idx = nucleus_sampling(logits, p=top_p)
            else:
                probs = F.softmax(logits, dim=0)
                sample_idx = torch.multinomial(probs, 1).item()

            char = chars[sample_idx]
            result += char

        return result


# =============================================================================
# Training
# =============================================================================

def train(
    model: HaikuRNN,
    dataset: HaikuDataset,
    device: torch.device,
    save_dir: str,
    num_epochs: int = 50,
    batch_size: int = 50,
    learning_rate: float = 0.002,
    decay_rate: float = 0.97,
    grad_clip: float = 5.0,
    save_every: int = 1000,
    log_every: int = 10
) -> HaikuRNN:
    """
    Train the model.

    Args:
        model: HaikuRNN model to train
        dataset: HaikuDataset with training data
        device: Device to train on
        save_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        decay_rate: Learning rate decay per epoch
        grad_clip: Gradient clipping threshold
        save_every: Save checkpoint every N batches
        log_every: Print loss every N batches

    Returns:
        Trained model
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save vocabulary
    chars, vocab, vocab_size = dataset.get_vocab_info()
    with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'wb') as f:
        pickle.dump((chars, vocab), f)

    # Save config
    config = {
        'vocab_size': vocab_size,
        'embed_size': model.embed_size,
        'hidden_size': model.hidden_size,
        'num_layers': model.num_layers,
        'model_type': model.model_type,
    }
    with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # MPS doesn't work well with multiprocessing
        drop_last=True
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    model.train()

    global_step = 0

    print(f"\nStarting training on {device}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}")
    print(f"Total batches per epoch: {len(dataloader)}")
    print("-" * 60)

    for epoch in range(num_epochs):
        # Decay learning rate
        current_lr = learning_rate * (decay_rate ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        epoch_loss = 0.0
        num_batches = 0
        hidden = None

        for batch_idx, (x, y) in enumerate(dataloader):
            start_time = time.time()

            x = x.to(device)
            y = y.to(device)

            # Initialize or detach hidden state
            if hidden is None:
                hidden = model.init_hidden(batch_size, device)
                if model.model_type != 'lstm':
                    hidden = hidden[0]
            else:
                hidden = model.detach_hidden(hidden) if model.model_type == 'lstm' else hidden.detach()

            # Forward pass
            optimizer.zero_grad()
            logits, hidden = model(x, hidden)

            # Compute loss
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            batch_time = time.time() - start_time

            # Logging
            if batch_idx % log_every == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Time: {batch_time:.3f}s")

            # Save checkpoint
            if global_step % save_every == 0:
                checkpoint_path = os.path.join(save_dir, f'model_step_{global_step}.pt')
                torch.save({
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'config': config,
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

        avg_loss = epoch_loss / num_batches
        print(f"\n=== Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f} ===\n")

        # Reset hidden state between epochs
        hidden = None

    # Save final model
    final_path = os.path.join(save_dir, 'model_final.pt')
    torch.save({
        'step': global_step,
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, final_path)
    print(f"Final model saved: {final_path}")

    return model


def load_model(
    save_dir: str,
    device: torch.device,
    checkpoint: Optional[str] = None
) -> Tuple[HaikuRNN, List[str], Dict[str, int]]:
    """
    Load a trained model from checkpoint.

    Args:
        save_dir: Directory containing saved model
        device: Device to load model to
        checkpoint: Optional specific checkpoint file, defaults to 'model_final.pt'

    Returns:
        Tuple of (model, chars, vocab)
    """
    # Load config
    config_path = os.path.join(save_dir, 'config.pkl')
    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    # Load vocabulary
    vocab_path = os.path.join(save_dir, 'chars_vocab.pkl')
    with open(vocab_path, 'rb') as f:
        chars, vocab = pickle.load(f)

    # Handle tuple chars (from old format)
    if isinstance(chars, tuple):
        chars = list(chars)

    # Create model
    model = HaikuRNN(
        vocab_size=config['vocab_size'],
        embed_size=config.get('embed_size', config.get('hidden_size', 128)),
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        model_type=config['model_type'],
    )

    # Load checkpoint
    if checkpoint is None:
        # Find the latest checkpoint
        checkpoint = os.path.join(save_dir, 'model_final.pt')
        if not os.path.exists(checkpoint):
            # Try to find any checkpoint
            checkpoints = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
            if checkpoints:
                checkpoint = os.path.join(save_dir, sorted(checkpoints)[-1])
            else:
                raise FileNotFoundError(f"No checkpoint found in {save_dir}")

    print(f"Loading checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()

    return model, chars, vocab


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PyTorch Haiku RNN - Train and generate Japanese haiku',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a new model:
    python model_pytorch.py train --data_dir data/issa-utf16 --save_dir save_pytorch

  Generate haiku:
    python model_pytorch.py sample --save_dir save_pytorch -n 500

  Generate with temperature:
    python model_pytorch.py sample --save_dir save_pytorch --temperature 0.8
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Training arguments
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data_dir', type=str, default='data/issa-utf16',
                              help='Data directory containing input.txt')
    train_parser.add_argument('--save_dir', type=str, default='save_pytorch',
                              help='Directory to save checkpoints')
    train_parser.add_argument('--model', type=str, default='lstm',
                              choices=['rnn', 'gru', 'lstm'],
                              help='RNN cell type')
    train_parser.add_argument('--embed_size', type=int, default=128,
                              help='Character embedding dimension')
    train_parser.add_argument('--hidden_size', type=int, default=128,
                              help='RNN hidden state size')
    train_parser.add_argument('--num_layers', type=int, default=2,
                              help='Number of RNN layers')
    train_parser.add_argument('--batch_size', type=int, default=50,
                              help='Training batch size')
    train_parser.add_argument('--seq_length', type=int, default=50,
                              help='Training sequence length')
    train_parser.add_argument('--num_epochs', type=int, default=50,
                              help='Number of training epochs')
    train_parser.add_argument('--learning_rate', type=float, default=0.002,
                              help='Initial learning rate')
    train_parser.add_argument('--decay_rate', type=float, default=0.97,
                              help='Learning rate decay per epoch')
    train_parser.add_argument('--grad_clip', type=float, default=5.0,
                              help='Gradient clipping threshold')
    train_parser.add_argument('--dropout', type=float, default=0.0,
                              help='Dropout probability')
    train_parser.add_argument('--save_every', type=int, default=1000,
                              help='Save checkpoint every N batches')
    train_parser.add_argument('--device', type=str, default=None,
                              choices=['mps', 'cuda', 'cpu'],
                              help='Device to use (auto-detected if not specified)')

    # Sampling arguments
    sample_parser = subparsers.add_parser('sample', help='Generate text from trained model')
    sample_parser.add_argument('--save_dir', type=str, default='save_pytorch',
                               help='Directory containing trained model')
    sample_parser.add_argument('--checkpoint', type=str, default=None,
                               help='Specific checkpoint file to load')
    sample_parser.add_argument('-n', '--num_chars', type=int, default=500,
                               help='Number of characters to generate')
    sample_parser.add_argument('--prime', type=str, default=' ',
                               help='Seed text for generation')
    sample_parser.add_argument('--temperature', type=float, default=1.0,
                               help='Sampling temperature (higher = more random)')
    sample_parser.add_argument('--sample', type=int, default=1,
                               choices=[0, 1, 2],
                               help='Sampling type: 0=greedy, 1=weighted, 2=conditional')
    sample_parser.add_argument('--top_k', type=int, default=None,
                               help='Use top-k sampling with k tokens')
    sample_parser.add_argument('--top_p', type=float, default=None,
                               help='Use nucleus sampling with probability p')
    sample_parser.add_argument('--device', type=str, default=None,
                               choices=['mps', 'cuda', 'cpu'],
                               help='Device to use (auto-detected if not specified)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Get device
    device = get_device(args.device)

    if args.command == 'train':
        # Create dataset
        dataset = HaikuDataset(
            args.data_dir,
            seq_length=args.seq_length
        )

        chars, vocab, vocab_size = dataset.get_vocab_info()

        # Create model
        model = HaikuRNN(
            vocab_size=vocab_size,
            embed_size=args.embed_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            model_type=args.model,
            dropout=args.dropout
        )

        print(f"\nModel architecture:")
        print(f"  Type: {args.model.upper()}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Embedding size: {args.embed_size}")
        print(f"  Hidden size: {args.hidden_size}")
        print(f"  Num layers: {args.num_layers}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train
        train(
            model=model,
            dataset=dataset,
            device=device,
            save_dir=args.save_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            decay_rate=args.decay_rate,
            grad_clip=args.grad_clip,
            save_every=args.save_every
        )

    elif args.command == 'sample':
        # Load model
        model, chars, vocab = load_model(args.save_dir, device, args.checkpoint)

        print(f"\nGenerating {args.num_chars} characters...")
        print(f"Temperature: {args.temperature}")
        print(f"Sampling type: {args.sample}")
        print("-" * 60)

        # Generate text
        if args.top_k is not None or args.top_p is not None:
            generated = sample_advanced(
                model=model,
                chars=chars,
                vocab=vocab,
                device=device,
                num_chars=args.num_chars,
                prime=args.prime,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
        else:
            generated = sample(
                model=model,
                chars=chars,
                vocab=vocab,
                device=device,
                num_chars=args.num_chars,
                prime=args.prime,
                temperature=args.temperature,
                sampling_type=args.sample
            )

        print(generated)


if __name__ == '__main__':
    main()
