"""
Byte-level tokenizer for LLaDA, based on ByT5's approach
"""

import torch
from typing import List, Optional, Union
import numpy as np

class ByteLevelTokenizer:
    """
    A simple byte-level tokenizer that works directly with UTF-8 bytes.
    No need for a vocabulary file since we work with raw bytes (0-255).
    """
    def __init__(self):
        self.vocab_size = 256  # Fixed vocabulary size for byte-level encoding
        self.bos_token = 1  # Beginning of sequence token
        self.eos_token = 2  # End of sequence token
        self.pad_token = 0  # Padding token
        self.unk_token = None  # No unknown token needed since we handle all bytes

    def encode(self, text: str, bos: bool = True, eos: bool = True) -> List[int]:
        """
        Encode a string into a list of byte ids.
        Args:
            text: The input text to encode
            bos: Whether to add beginning of sequence token
            eos: Whether to add end of sequence token
        Returns:
            A list of integers representing byte values
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Convert string to bytes and get integer values
        byte_values = list(text.encode('utf-8'))
        
        # Add special tokens if requested
        if bos:
            byte_values = [self.bos_token] + byte_values
        if eos:
            byte_values = byte_values + [self.eos_token]
            
        return byte_values

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of byte ids back into a string.
        Args:
            token_ids: List of integer token ids
            skip_special_tokens: Whether to remove special tokens from output
        Returns:
            Decoded string
        """
        if skip_special_tokens:
            # Filter out special tokens
            bytes_list = [b for b in token_ids 
                         if b not in {self.bos_token, self.eos_token, self.pad_token}]
        else:
            bytes_list = token_ids
            
        try:
            # Convert bytes back to string
            return bytes(bytes_list).decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Warning: Error decoding bytes: {e}")
            return ""

    def encode_batch(self, texts: List[str], max_length: Optional[int] = None,
                    pad: bool = True, bos: bool = True, eos: bool = True) -> torch.Tensor:
        """
        Encode a batch of texts into a padded tensor of byte ids.
        Args:
            texts: List of input strings to encode
            max_length: Maximum sequence length (will truncate if necessary)
            pad: Whether to pad sequences to max_length
            bos: Whether to add beginning of sequence token
            eos: Whether to add end of sequence token
        Returns:
            Tensor of shape (batch_size, max_length) containing byte ids
        """
        # Encode all texts
        encoded = [self.encode(text, bos=bos, eos=eos) for text in texts]
        
        if max_length is None:
            # Use length of longest sequence
            max_length = max(len(seq) for seq in encoded)
            
        # Initialize output tensor with padding token
        batch_size = len(texts)
        output = torch.full((batch_size, max_length), self.pad_token, dtype=torch.long)
        
        # Fill in encoded sequences
        for i, seq in enumerate(encoded):
            length = min(len(seq), max_length)
            output[i, :length] = torch.tensor(seq[:length])
            
        return output

    def decode_batch(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of token ids back into strings.
        Args:
            token_ids: Tensor of shape (batch_size, seq_len) containing byte ids
            skip_special_tokens: Whether to remove special tokens from output
        Returns:
            List of decoded strings
        """
        if token_ids.dim() != 2:
            raise ValueError("Expected 2D tensor input of shape (batch_size, seq_len)")
            
        return [self.decode(ids.tolist(), skip_special_tokens=skip_special_tokens) 
                for ids in token_ids]

    def get_vocab(self) -> dict:
        """
        Return the vocabulary mapping.
        For byte-level tokenization, this is simply the mapping of bytes 0-255 to themselves,
        plus the special tokens.
        """
        vocab = {
            '<pad>': self.pad_token,
            '<bos>': self.bos_token,
            '<eos>': self.eos_token,
        }
        # Add byte mappings
        for i in range(256):
            if i not in {self.pad_token, self.bos_token, self.eos_token}:
                vocab[f'<byte_{i}>'] = i
        return vocab
