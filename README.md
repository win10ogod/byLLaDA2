# LLaDA: Large Language Diffusion with Masking

This is a PyTorch implementation of the LLaDA architecture, based on llama2.c but enhanced with:

1. **Diffusion-based Training** - Uses diffusion-based masked language modeling approach for more flexible learning
2. **Byte-level Encoding** - Direct byte-level modeling using ByT5's approach, removing the need for traditional tokenization
3. **Semi-autoregressive Generation** - More efficient generation through block-wise prediction

## Features

- **Token-free Operation**: Works directly with UTF-8 bytes, supporting any language without tokenization
- **Diffusion Masking**: Novel training approach using diffusion-based masking schedule
- **Efficient Architecture**: Based on the efficient llama2.c implementation
- **Universal Language Support**: Handles any language or script through byte-level processing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```python
from model import ModelArgs, LLaDA
from tokenizer import ByteLevelTokenizer

# Initialize tokenizer
tokenizer = ByteLevelTokenizer()

# Configure model
config = ModelArgs(
    dim=1024,
    n_layers=12,
    n_heads=16,
    vocab_size=256,  # Fixed for byte-level encoding
    max_seq_len=2048
)

# Create model
model = LLaDA(config)

# Training example
tokens = tokenizer.encode("Your training text")
targets = tokens[1:]  # Shift right for next token prediction
output = model(tokens, targets)
```

### Generation

```python
prompt = "Once upon a time"
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens])

# Generate text
outputs = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40
)

# Decode output
generated_text = tokenizer.decode(outputs[0].tolist())
print(generated_text)
```

## Architecture Details

### LLaDA Components:

1. **Byte-Level Tokenization**
   - Works directly with UTF-8 bytes (256 vocabulary size)
   - No tokenizer training needed
   - Universal language support

2. **Diffusion-based Masking**
   - Cosine masking schedule
   - Dynamic mask ratio during training
   - Improved bidirectional context understanding

3. **Semi-autoregressive Generation**
   - Block-wise parallel generation
   - Efficient inference through masked prediction

## Key Differences from llama2.c

1. **Tokenization**
   - Replaced SentencePiece tokenizer with byte-level encoding
   - Removed need for vocabulary files
   - Simplified preprocessing

2. **Training**
   - Added diffusion-based masking mechanism
   - Modified attention to support masked tokens
   - Enhanced bidirectional learning

3. **Generation**
   - Added semi-autoregressive generation
   - Improved efficiency through block generation
   - Maintained compatibility with original inference code

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Based on [llama2.c](https://github.com/karpathy/llama2.c)
- Inspired by ByT5's byte-level processing
- Built with PyTorch
- https://arxiv.org/abs/2502.09992 (I am not the author of the paper. All credit goes to the author of the paper.)
