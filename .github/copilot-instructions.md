# Copilot Instructions for Build-an-LLM-from-beginning

## Project Overview

This is an educational project to implement Large Language Model (LLM) components from scratch using PyTorch. The project follows a chapter-based structure where each chapter explores a specific aspect of building an LLM.

## Environment & Setup

**Python Version:** 3.12 or higher (specified in `.python-version`)

**Dependency Management:** Uses `uv` for fast Python package management
- Install dependencies: `uv pip install -e .` (or use your system pip)
- Dependencies are defined in `pyproject.toml`

**Key Dependencies:**
- `torch` (≥2.12.1) - Neural network framework
- `tiktoken` (≥0.13.0) - Byte-pair encoding tokenizer for text processing
- `numpy` (≥2.4.6) - Numerical computing
- `matplotlib` & `pandas` - Data visualization and analysis

## Project Structure

```
.
├── src/                          # Core implementations organized by chapter
│   ├── ch02_text_data.py        # Text data processing
│   ├── ch02_02_bytePairEncoding.py  # BPE tokenization implementation
│   ├── ch02_03_slidingWindowSampling.py  # Sampling strategy
│   └── ch04_implement_gpt.py    # Full GPT model architecture
├── test/                         # Example/demonstration scripts (not pytest)
│   ├── test_mlp.py              # MLP demonstration
│   ├── test_training_loop.py    # Training loop examples
│   ├── test_dataloader.py       # DataLoader usage patterns
│   └── test_single_gpu.py       # Single GPU training
├── build-llm-from-beginning.ipynb  # Jupyter notebook with chapter-by-chapter breakdown
└── main.py                       # Entry point
```

## Running the Code

**Run a single example/demo script:**
```bash
python test/test_mlp.py
python test/test_training_loop.py
python src/ch04_implement_gpt.py
```

**Run the Jupyter notebook:**
```bash
jupyter notebook build-llm-from-beginning.ipynb
```

**Main entry point:**
```bash
python main.py
```

## Key Architectural Patterns

### Model Configuration as Dictionary
Models use configuration dictionaries (not dataclasses or separate config files). This allows easy parameter experimentation:

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

### Model Class Pattern
All models inherit from `torch.nn.Module`. When creating new models:
1. Define config dict with architecture parameters
2. Pass config through `__init__` layers
3. Implement `forward()` method that transforms input
4. Use nested blocks (e.g., `nn.Sequential` of transformer blocks)

### Dataset & DataLoader Usage
Custom PyTorch `Dataset` classes are used to structure training data:
```python
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    def __len__(self):
        return len(self.labels)
```

## Code Conventions

1. **Tokenization**: Use `tiktoken.get_encoding("gpt2")` for GPT2-compatible tokenization
2. **Device Handling**: Explicitly move tensors to device with `.to()` or `.device` when needed
3. **Seed Management**: Use `torch.manual_seed()` for reproducibility
4. **Batch Processing**: Always ensure proper batch dimensions:
   - Input: `(batch_size, sequence_length)`
   - Output logits: `(batch_size, sequence_length, vocab_size)`
5. **Forward Passes**: Use `torch.no_grad()` context for inference to disable gradient tracking

## Testing & Validation

The `test/` directory contains demonstration/example scripts that show proper usage patterns rather than pytest unit tests. These scripts are meant to be run directly to validate implementations:

```bash
python test/test_training_loop.py  # Validate training loop works
python test/test_single_gpu.py     # Validate single GPU training
```

When adding new components, create example scripts in `test/` that demonstrate proper usage and validate correctness.

## Common Tasks

**Adding a new chapter/module:**
1. Create `src/chXX_description.py` following existing chapter naming
2. Start with config dict if it's a model
3. Add corresponding example in `test/test_feature.py`

**Training a model:**
- See `test/test_training_loop.py` for complete training loop pattern
- Use torch DataLoader for batch processing
- Remember to set seeds for reproducibility

**Working with different model sizes:**
- Modify the config dict parameters (vocab_size, n_layers, emb_dim, etc.)
- Pass the modified config to model initialization
- Configs can be created on-the-fly for experimentation
