# Build an LLM from Beginning

An educational project implementing Large Language Model (LLM) components from scratch using PyTorch. This project progressively builds the fundamental components of GPT-style transformers, from basic neural networks to a complete multi-head attention transformer architecture.

## Overview

This project is organized as a series of chapters, each exploring a specific component or concept in building an LLM:

- **Chapter 2**: Text data processing and tokenization (Byte-Pair Encoding)
- **Chapter 3**: Multi-head attention mechanism
- **Chapter 4**: Full GPT transformer model implementation

Each chapter includes implementation files (`src/chXX_*.py`) and corresponding test/demonstration scripts (`test/test_*.py`) that show how to use the components.

## Project Structure

```
.
├── src/                              # Core implementations
│   ├── ch02_text_data.py            # Text data preprocessing
│   ├── ch02_02_bytePairEncoding.py  # BPE tokenizer implementation
│   ├── ch02_03_slidingWindowSampling.py  # Text sampling strategy
│   ├── ch03_MultiHeadAttention.py   # Multi-head attention layer
│   ├── ch04_gpt.py                  # GPT model with transformer blocks
│   ├── ch04_transformer.py          # Transformer blocks
│   └── ch04_implement_gpt.py        # GPT model architecture
├── test/                             # Example scripts and demonstrations
│   ├── test_pytorch.py              # PyTorch basics
│   ├── test_auto_diff.py            # Autograd/backpropagation
│   ├── test_computation_graph.py    # Computational graphs
│   ├── test_mlp.py                  # Multi-layer perceptron
│   ├── test_batch_layer_normalization.py  # Layer normalization
│   ├── test_shortcut_connection.py  # Residual connections
│   ├── test_dataloader.py           # Data loading patterns
│   ├── test_training_loop.py        # Training loop examples
│   ├── test_single_gpu.py           # Single GPU training
│   └── test_multi_gpus.py           # Multi-GPU training
├── build-llm-from-beginning.ipynb   # Comprehensive Jupyter notebook
├── pyproject.toml                   # Project dependencies
└── main.py                          # Entry point

```

## Requirements

- **Python**: 3.12 or higher (see `.python-version`)
- **Package Manager**: `uv` (recommended) or `pip`

## Installation

```bash
# Using uv
uv pip install -e .

# Using pip
pip install -e .
```

### Dependencies

Core dependencies (from `pyproject.toml`):
- `torch>=2.12.1` - Neural network framework
- `tiktoken>=0.13.0` - GPT2-compatible tokenizer
- `numpy>=2.4.6` - Numerical computing
- `matplotlib>=3.11.0` - Visualization
- `pandas>=3.0.3` - Data manipulation

## Quick Start

### Run Example Scripts

Each test file demonstrates a specific concept:

```bash
# PyTorch basics and neural networks
python test/test_pytorch.py
python test/test_mlp.py

# Backpropagation and autograd
python test/test_auto_diff.py
python test/test_computation_graph.py

# Transformer components
python test/test_batch_layer_normalization.py
python test/test_shortcut_connection.py

# Data loading
python test/test_dataloader.py

# Training loops
python test/test_training_loop.py
python test/test_single_gpu.py

# GPT implementation
python src/ch04_implement_gpt.py
```

### Run the Jupyter Notebook

For a comprehensive walkthrough with detailed explanations:

```bash
jupyter notebook build-llm-from-beginning.ipynb
```

### Run Main Entry Point

```bash
python main.py
```

## Key Concepts Covered

### Text Processing (Chapter 2)
- Raw text preprocessing
- Byte-Pair Encoding (BPE) tokenization using `tiktoken`
- Sliding window sampling for sequence creation

### Attention Mechanism (Chapter 3)
- Multi-head self-attention implementation
- Query, Key, Value projections
- Causal masking for autoregressive generation
- Attention weight computation and dropout

### Transformer Architecture (Chapter 4)
- Complete GPT model implementation
- Transformer blocks with attention and feed-forward layers
- Token and positional embeddings
- Training procedures for LLMs

### Foundational Concepts (Tests)
- Automatic differentiation (autograd)
- Computational graphs
- Backpropagation
- Batch processing with DataLoaders
- Multi-GPU training patterns

## Architecture Patterns

### Model Configuration
Models use configuration dictionaries for flexibility:

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

### PyTorch Conventions
- Models inherit from `torch.nn.Module`
- Use `torch.manual_seed()` for reproducibility
- Apply `torch.no_grad()` context for inference
- Explicit device management with `.to()`

## Usage Conventions

1. **Tokenization**: Use `tiktoken.get_encoding("gpt2")` for token operations
2. **Batch Dimensions**: Input shape `(batch_size, sequence_length)` → output `(batch_size, sequence_length, vocab_size)`
3. **Device Handling**: Always manage device placement explicitly
4. **Seeds**: Set seeds before model initialization for reproducibility

## Example: Training a Simple Model

```python
import torch
from src.ch04_gpt import GPT, GPT_CONFIG_124M

# Set random seed for reproducibility
torch.manual_seed(42)

# Initialize model
model = GPT(GPT_CONFIG_124M)

# Create dummy input
batch_size, seq_length = 2, 128
input_ids = torch.randint(0, 50257, (batch_size, seq_length))

# Forward pass
logits = model(input_ids)
print(f"Output shape: {logits.shape}")  # (batch_size, seq_length, vocab_size)
```

## Learning Resources

This project is designed for educational purposes. Each chapter builds upon previous concepts:

1. Start with foundational PyTorch concepts (`test/test_pytorch.py`)
2. Understand backpropagation (`test/test_auto_diff.py`)
3. Learn basic neural networks (`test/test_mlp.py`)
4. Explore attention mechanisms (`src/ch03_MultiHeadAttention.py`)
5. Implement a complete GPT model (`src/ch04_gpt.py`)

The Jupyter notebook (`build-llm-from-beginning.ipynb`) provides a comprehensive walkthrough of the entire process with detailed explanations.

## Notes

- This is an educational implementation focused on clarity and understanding, not production optimization
- Example scripts in `test/` are meant to be run directly to validate implementations
- GPU support is demonstrated in `test/test_single_gpu.py` and `test/test_multi_gpus.py`
- Results are saved to `model.pth` for checkpoint management

## License

See LICENSE file for details.

## References

This project implements concepts from transformer-based language models, particularly those found in:
- GPT (Generative Pre-trained Transformer)
- Attention is All You Need (Vaswani et al., 2017)