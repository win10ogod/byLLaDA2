"""
Configuration settings for LLaDA models of different sizes
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    name: str
    dim: int
    n_layers: int
    n_heads: int
    max_seq_len: int
    dropout: float
    vocab_size: int = 256  # Fixed for byte-level encoding
    multiple_of: int = 256
    norm_eps: float = 1e-5

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_iters: int
    lr_decay_iters: int
    min_lr: float
    grad_clip: float
    grad_accumulation_steps: int
    mask_schedule: str = 'cosine'
    base_mask_ratio: float = 0.15

# Tiny model for testing
tiny_config = ModelConfig(
    name="llada-tiny",
    dim=512,
    n_layers=6,
    n_heads=8,
    max_seq_len=1024,
    dropout=0.1
)

# Small model (~125M parameters)
small_config = ModelConfig(
    name="llada-small",
    dim=768,
    n_layers=12,
    n_heads=12,
    max_seq_len=2048,
    dropout=0.1
)

# Base model (~350M parameters)
base_config = ModelConfig(
    name="llada-base",
    dim=1024,
    n_layers=24,
    n_heads=16,
    max_seq_len=2048,
    dropout=0.1
)

# Large model (~750M parameters)
large_config = ModelConfig(
    name="llada-large",
    dim=1536,
    n_layers=32,
    n_heads=24,
    max_seq_len=2048,
    dropout=0.1
)

# Training configurations for different scenarios
default_training = TrainingConfig(
    batch_size=128,
    learning_rate=5e-4,
    weight_decay=1e-1,
    warmup_iters=1000,
    lr_decay_iters=100000,
    min_lr=5e-5,
    grad_clip=1.0,
    grad_accumulation_steps=1
)

# Training config for smaller batches (useful for limited memory)
small_batch_training = TrainingConfig(
    batch_size=32,
    learning_rate=3e-4,
    weight_decay=1e-1,
    warmup_iters=2000,
    lr_decay_iters=150000,
    min_lr=5e-5,
    grad_clip=1.0,
    grad_accumulation_steps=4
)

# Training config for distributed training
distributed_training = TrainingConfig(
    batch_size=256,
    learning_rate=8e-4,
    weight_decay=1e-1,
    warmup_iters=1000,
    lr_decay_iters=80000,
    min_lr=5e-5,
    grad_clip=1.0,
    grad_accumulation_steps=1
)

MODEL_CONFIGS = {
    'tiny': tiny_config,
    'small': small_config,
    'base': base_config,
    'large': large_config
}

TRAINING_CONFIGS = {
    'default': default_training,
    'small_batch': small_batch_training,
    'distributed': distributed_training
}

def get_model_config(name: str) -> ModelConfig:
    """Get model configuration by name"""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model config: {name}. Available configs: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[name]

def get_training_config(name: str) -> TrainingConfig:
    """Get training configuration by name"""
    if name not in TRAINING_CONFIGS:
        raise ValueError(f"Unknown training config: {name}. Available configs: {list(TRAINING_CONFIGS.keys())}")
    return TRAINING_CONFIGS[name]

# Example usage:
if __name__ == "__main__":
    # Get model and training configs
    model_cfg = get_model_config('base')
    train_cfg = get_training_config('default')
    
    print(f"Model config '{model_cfg.name}':")
    print(f"  Dimension: {model_cfg.dim}")
    print(f"  Layers: {model_cfg.n_layers}")
    print(f"  Heads: {model_cfg.n_heads}")
    print(f"  Max sequence length: {model_cfg.max_seq_len}")
    
    print(f"\nTraining config:")
    print(f"  Batch size: {train_cfg.batch_size}")
    print(f"  Learning rate: {train_cfg.learning_rate}")
    print(f"  Warmup iterations: {train_cfg.warmup_iters}")
