"""
Dataset loading and preprocessing for LLaDA
Supports byte-level processing of text data
"""

import os
import json
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from tokenizer import ByteLevelTokenizer

class TextDataset(Dataset):
    """Dataset for byte-level text processing"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 max_seq_len: int = 256,
                 tokenizer: Optional[ByteLevelTokenizer] = None):
        self.data_dir = data_dir
        self.split = split
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer or ByteLevelTokenizer()
        
        # Load and preprocess data
        self.examples = self.load_data()
    
    def load_data(self) -> List[np.ndarray]:
        """Load and preprocess text data into byte sequences"""
        examples = []
        
        # Get all .txt or .json files in data directory
        files = []
        for ext in ['*.txt', '*.json']:
            files.extend(glob.glob(os.path.join(self.data_dir, self.split, ext)))
            
        if not files:
            raise ValueError(f"No data files found in {os.path.join(self.data_dir, self.split)}")
            
        print(f"Loading {self.split} data from {len(files)} files...")
        
        for file_path in tqdm(files):
            if file_path.endswith('.json'):
                # Handle JSON files (assume each line is a JSON object with 'text' field)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            text = data.get('text', '')
                            if text:
                                # Convert text to byte sequence
                                byte_ids = self.tokenizer.encode(text, bos=True, eos=True)
                                if len(byte_ids) > 1:  # Skip empty sequences
                                    examples.append(np.array(byte_ids))
                        except json.JSONDecodeError:
                            continue
            else:
                # Handle plain text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    # Split into manageable chunks
                    chunks = self.chunk_text(text)
                    for chunk in chunks:
                        byte_ids = self.tokenizer.encode(chunk, bos=True, eos=True)
                        if len(byte_ids) > 1:  # Skip empty sequences
                            examples.append(np.array(byte_ids))
        
        print(f"Loaded {len(examples)} examples")
        return examples
    
    def chunk_text(self, text: str, min_chunk_size: int = 100) -> List[str]:
        """Split text into chunks at sentence boundaries"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Simple sentence splitting on .!?
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_bytes = len(sentence.encode('utf-8'))
            
            if current_length + sentence_bytes > self.max_seq_len:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_bytes
            else:
                current_chunk.append(sentence)
                current_length += sentence_bytes
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training example"""
        example = self.examples[idx]
        
        # Handle sequences longer than max_seq_len
        if len(example) > self.max_seq_len:
            # Randomly select a position to start
            start_idx = np.random.randint(0, len(example) - self.max_seq_len)
            example = example[start_idx:start_idx + self.max_seq_len]
        
        # Convert to tensor
        x = torch.from_numpy(example[:-1]).long()  # Input sequence
        y = torch.from_numpy(example[1:]).long()   # Target sequence (shifted by 1)
        
        return {'input_ids': x, 'labels': y}

def create_dataloaders(
    data_dir: str,
    batch_size: int,
    max_seq_len: int,
    num_workers: int = 4,
    tokenizer: Optional[ByteLevelTokenizer] = None
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders"""
    
    # Create datasets
    train_dataset = TextDataset(
        data_dir=data_dir,
        split='train',
        max_seq_len=max_seq_len,
        tokenizer=tokenizer
    )
    
    val_dataset = TextDataset(
        data_dir=data_dir,
        split='val',
        max_seq_len=max_seq_len,
        tokenizer=tokenizer
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, val_loader

# Example usage:
if __name__ == "__main__":
    # Test dataset loading
    tokenizer = ByteLevelTokenizer()
    dataset = TextDataset("data/", split="train", max_seq_len=256, tokenizer=tokenizer)
    print(f"Dataset size: {len(dataset)}")
    
    # Test batch loading
    train_loader, val_loader = create_dataloaders(
        data_dir="data/",
        batch_size=32,
        max_seq_len=256,
        tokenizer=tokenizer
    )
    
    # Inspect a batch
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
