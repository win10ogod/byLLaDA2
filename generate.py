"""
Text generation script for LLaDA model
"""

import os
import torch
import argparse
from contextlib import nullcontext
from typing import List, Optional
import torch.nn.functional as F
from model import ModelArgs, LLaDA
from tokenizer import ByteLevelTokenizer
from config import get_model_config, MODEL_CONFIGS

def get_args():
    parser = argparse.ArgumentParser(description='Generate text with LLaDA model')
    
    # Model configuration
    parser.add_argument('--model_size', type=str, default='base', choices=list(MODEL_CONFIGS.keys()),
                      help='Model size configuration')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    
    # Generation parameters
    parser.add_argument('--max_new_tokens', type=int, default=100,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                      help='Sampling temperature (0.0 = greedy)')
    parser.add_argument('--top_k', type=int, default=40,
                      help='Top-k sampling (0 = disabled)')
    parser.add_argument('--top_p', type=float, default=0.9,
                      help='Top-p sampling (1.0 = disabled)')
    parser.add_argument('--num_return_sequences', type=int, default=1,
                      help='Number of sequences to generate')
    
    # Input/Output
    parser.add_argument('--prompt', type=str, default='',
                      help='Text prompt for generation')
    parser.add_argument('--input_file', type=str, default='',
                      help='File containing prompts (one per line)')
    parser.add_argument('--output_file', type=str, default='',
                      help='File to save generated text')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                      choices=['float32', 'bfloat16', 'float16'],
                      help='Data type for inference')
    
    return parser.parse_args()

def load_model(args):
    """Load model from checkpoint"""
    print(f"Loading model from {args.checkpoint}")
    
    # Get model configuration
    model_config = get_model_config(args.model_size)
    
    # Initialize model
    model_args = ModelArgs(
        dim=model_config.dim,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        vocab_size=model_config.vocab_size,
        max_seq_len=model_config.max_seq_len,
        dropout=0.0  # No dropout during inference
    )
    
    model = LLaDA(model_args)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state'])
    
    # Move model to device and set to eval mode
    model.to(args.device)
    model.eval()
    
    return model

def generate_text(
    model: LLaDA,
    tokenizer: ByteLevelTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    top_p: float = 0.9,
    num_return_sequences: int = 1,
    ctx: nullcontext = nullcontext(),
) -> List[str]:
    """Generate text from prompt using diffusion process"""
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt)
    prompt_ids = torch.tensor(prompt_ids).unsqueeze(0).to(model.device)
    
    # Generate multiple sequences
    generated_sequences = []
    
    for _ in range(num_return_sequences):
        with torch.no_grad(), ctx:
            # Start with fully masked sequence
            masked_ids = torch.full((1, max_new_tokens), 
                                  model.tok_embeddings.weight.shape[0] - 1,  # Mask token
                                  device=model.device)
            
            # Concatenate prompt and masked sequence
            input_ids = torch.cat([prompt_ids, masked_ids], dim=1)
            
            # Diffusion generation process
            for t in torch.linspace(1, 0, max_new_tokens, device=model.device):
                # Predict all tokens
                logits = model(input_ids)
                
                # Apply temperature and top-k filtering
                logits = logits[:, -max_new_tokens:] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Sample next tokens
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                
                # Update input_ids with predicted tokens
                input_ids[:, -max_new_tokens:] = next_tokens
            
            # Decode generated sequence
            generated_text = tokenizer.decode(input_ids[0, len(prompt_ids[0]):].tolist())
            generated_sequences.append(generated_text)
    
    return generated_sequences

def main():
    args = get_args()
    
    # Setup device and dtype
    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Load model and tokenizer
    model = load_model(args)
    tokenizer = ByteLevelTokenizer()
    
    # Get prompts
    prompts = []
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            prompts.extend(line.strip() for line in f if line.strip())
    if args.prompt:
        prompts.append(args.prompt)
    if not prompts:
        prompts.append("Once upon a time")  # Default prompt
    
    # Open output file if specified
    out_file = open(args.output_file, 'w', encoding='utf-8') if args.output_file else None
    
    # Generate text for each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"\nGenerating text for prompt {i}/{len(prompts)}:")
        print(f"Prompt: {prompt}")
        
        generated_sequences = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_return_sequences,
            ctx=ctx
        )
        
        # Print and/or save generated sequences
        for j, sequence in enumerate(generated_sequences, 1):
            output = f"\nGenerated sequence {j}:\n{sequence}\n"
            print(output)
            
            if out_file:
                out_file.write(f"Prompt {i}: {prompt}\n")
                out_file.write(output)
                out_file.write("-" * 80 + "\n")
    
    if out_file:
        out_file.close()

if __name__ == "__main__":
    main()
