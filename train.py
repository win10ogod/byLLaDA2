"""
Training script for LLaDA (Large Language Diffusion with Masking)
"""

import os
import time
import math
import wandb
import torch
import argparse
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

from model import ModelArgs, LLaDA
from tokenizer import ByteLevelTokenizer
from dataset import create_dataloaders
from config import get_model_config, get_training_config, MODEL_CONFIGS, TRAINING_CONFIGS

def get_args():
    parser = argparse.ArgumentParser(description='Train LLaDA model')
    
    # Model and training configuration
    parser.add_argument('--model_size', type=str, default='base', choices=list(MODEL_CONFIGS.keys()),
                      help='Model size configuration')
    parser.add_argument('--train_config', type=str, default='default', choices=list(TRAINING_CONFIGS.keys()),
                      help='Training configuration')
    
    # I/O and logging
    parser.add_argument('--out_dir', type=str, default='out',
                      help='Output directory for checkpoints')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Data directory')
    parser.add_argument('--eval_interval', type=int, default=1000,
                      help='How often to evaluate')
    parser.add_argument('--log_interval', type=int, default=1,
                      help='How often to log training progress')
    parser.add_argument('--eval_iters', type=int, default=100,
                      help='Number of iterations for evaluation')
    parser.add_argument('--always_save_checkpoint', action='store_true',
                      help='Always save checkpoint after evaluation')
    parser.add_argument('--init_from', type=str, default='scratch',
                      help='Init from scratch or load from checkpoint')
    
    # Wandb logging
    parser.add_argument('--wandb', action='store_true',
                      help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='llada',
                      help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='',
                      help='Wandb run name')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                      choices=['float32', 'bfloat16', 'float16'],
                      help='Data type for training')
    parser.add_argument('--compile', action='store_true',
                      help='Use torch.compile')
    
    return parser.parse_args()

def get_lr(it: int, warmup_iters: int, lr_decay_iters: int, learning_rate: float, min_lr: float) -> float:
    """Get learning rate based on iteration"""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def train():
    # Get command line arguments and configurations
    args = get_args()
    model_config = get_model_config(args.model_size)
    train_config = get_training_config(args.train_config)
    
    # Initialize distributed training if needed
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        args.device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(args.device)
        master_process = ddp_rank == 0
        train_config.batch_size = train_config.batch_size // ddp_world_size
    else:
        master_process = True
        ddp_world_size = 1
    
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Initialize tokenizer and model
    tokenizer = ByteLevelTokenizer()
    
    model_args = ModelArgs(
        dim=model_config.dim,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        vocab_size=model_config.vocab_size,
        max_seq_len=model_config.max_seq_len,
        dropout=model_config.dropout,
    )
    
    model = LLaDA(model_args)
    model.to(args.device)
    
    # Load checkpoint if specified
    if args.init_from != 'scratch':
        print(f"Loading model from {args.init_from}")
        checkpoint = torch.load(args.init_from, map_location=args.device)
        model.load_state_dict(checkpoint['model_state'])
    
    # Compile model if requested
    if args.compile and device_type == 'cuda':
        print("Compiling model (this may take a few minutes)...")
        model = torch.compile(model)
    
    # Wrap model in DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # Create optimizer
    optimizer = model.configure_optimizers(
        train_config.weight_decay,
        train_config.learning_rate,
        (0.9, 0.95),  # betas
        device_type
    )
    
    # Load dataset
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=train_config.batch_size,
        max_seq_len=model_config.max_seq_len,
        num_workers=4,
        tokenizer=tokenizer
    )
    
    if ddp:
        train_loader.sampler = DistributedSampler(train_loader.dataset)
    
    # Initialize wandb
    if args.wandb and master_process:
        wandb_run_name = args.wandb_run_name or f"llada-{model_config.name}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config={
                'model_config': vars(model_config),
                'train_config': vars(train_config),
                'args': vars(args)
            }
        )
    
    # Training loop
    best_val_loss = float('inf')
    iter_num = 0
    
    model.train()
    for epoch in range(10000):  # Large number, we'll break based on iter_num
        if ddp:
            train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(train_loader):
            t0 = time.time()
            
            # Place data on device
            input_ids = batch['input_ids'].to(args.device)
            labels = batch['labels'].to(args.device)
            
            # Determine and set the learning rate for this iteration
            lr = get_lr(
                iter_num,
                train_config.warmup_iters,
                train_config.lr_decay_iters,
                train_config.learning_rate,
                train_config.min_lr
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass
            with ctx:
                logits = model(input_ids, labels)
                loss = model.last_loss
                loss = loss / train_config.grad_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if gradient accumulation is done
            if (batch_idx + 1) % train_config.grad_accumulation_steps == 0:
                if train_config.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                iter_num += 1
            
                # Logging
                if iter_num % args.log_interval == 0 and master_process:
                    t1 = time.time()
                    dt = t1 - t0
                    print(f"iter {iter_num}: loss {loss.item()*train_config.grad_accumulation_steps:.4f}, time {dt*1000:.2f}ms")
                    
                    if args.wandb:
                        wandb.log({
                            'iter': iter_num,
                            'loss': loss.item() * train_config.grad_accumulation_steps,
                            'lr': lr,
                        })
                
                # Evaluation
                if iter_num > 0 and iter_num % args.eval_interval == 0 and master_process:
                    val_loss = evaluate(model, val_loader, ctx)
                    print(f"step {iter_num}: val loss {val_loss:.4f}")
                    
                    if args.wandb:
                        wandb.log({
                            'val_loss': val_loss,
                        })
                    
                    # Save checkpoint if validation loss improved
                    if val_loss < best_val_loss or args.always_save_checkpoint:
                        best_val_loss = val_loss
                        if iter_num > 0:
                            checkpoint = {
                                'model_args': model_args,
                                'model_state': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'iter_num': iter_num,
                                'best_val_loss': best_val_loss,
                                'config': {
                                    'model': vars(model_config),
                                    'train': vars(train_config),
                                    'args': vars(args)
                                }
                            }
                            checkpoint_dir = os.path.join(args.out_dir, model_config.name)
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            print(f"saving checkpoint to {checkpoint_dir}")
                            torch.save(checkpoint, os.path.join(checkpoint_dir, 'ckpt.pt'))
                    
                    model.train()
            
            if iter_num >= train_config.lr_decay_iters:
                break
        
        if iter_num >= train_config.lr_decay_iters:
            break
    
    # Clean up
    if ddp:
        destroy_process_group()

@torch.no_grad()
def evaluate(model, val_loader, ctx):
    """Evaluate model on validation set"""
    model.eval()
    losses = []
    
    for batch in val_loader:
        input_ids = batch['input_ids'].to(model.device)
        labels = batch['labels'].to(model.device)
        
        with ctx:
            logits = model(input_ids, labels)
            loss = model.last_loss
        
        losses.append(loss.item())
    
    return torch.tensor(losses).mean()

if __name__ == "__main__":
    train()
