"""
Test script to verify LLaDA model components and setup
"""

import os
import torch
import unittest
from typing import Tuple

from model import ModelArgs, LLaDA
from tokenizer import ByteLevelTokenizer
from dataset import TextDataset, create_dataloaders
from config import get_model_config, get_training_config

class TestLLaDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cls.model_config = get_model_config('tiny')  # Use tiny model for testing
        cls.train_config = get_training_config('small_batch')
        cls.tokenizer = ByteLevelTokenizer()
        
        # Create a small test dataset
        os.makedirs('test_data/train', exist_ok=True)
        os.makedirs('test_data/val', exist_ok=True)
        cls._create_test_data()
        
        # Initialize model
        cls.model = cls._create_model()

    @classmethod
    def _create_model(cls) -> LLaDA:
        """Create test model instance"""
        model_args = ModelArgs(
            dim=cls.model_config.dim,
            n_layers=cls.model_config.n_layers,
            n_heads=cls.model_config.n_heads,
            vocab_size=cls.model_config.vocab_size,
            max_seq_len=cls.model_config.max_seq_len,
            dropout=cls.model_config.dropout
        )
        model = LLaDA(model_args)
        model.to(cls.device)
        return model

    @classmethod
    def _create_test_data(cls):
        """Create test dataset files"""
        train_text = "This is a test sentence for training. Here is another sentence.\n"
        val_text = "This is a test sentence for validation. Here is another sentence.\n"
        
        with open('test_data/train/test.txt', 'w', encoding='utf-8') as f:
            f.write(train_text * 10)
        with open('test_data/val/test.txt', 'w', encoding='utf-8') as f:
            f.write(val_text * 5)

    def test_tokenizer(self):
        """Test tokenizer functionality"""
        text = "Hello, world!"
        tokens = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(tokens)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertEqual(text, decoded.rstrip())
        
        # Test batch encoding
        texts = ["Hello", "World"]
        batch = self.tokenizer.encode_batch(texts, max_length=10)
        self.assertEqual(batch.shape[0], 2)
        self.assertEqual(batch.shape[1], 10)

    def test_model_forward(self):
        """Test model forward pass"""
        batch_size = 2
        seq_len = 16
        
        # Create random input
        input_ids = torch.randint(0, 256, (batch_size, seq_len)).to(self.device)
        labels = torch.randint(0, 256, (batch_size, seq_len)).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(input_ids, labels)
        
        self.assertEqual(logits.shape, (batch_size, seq_len, 256))
        self.assertIsNotNone(self.model.last_loss)
        self.assertTrue(torch.isfinite(self.model.last_loss))

    def test_model_generation(self):
        """Test text generation"""
        prompt = "Once upon a time"
        input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=20,
                temperature=0.8,
                top_k=40
            )
        
        self.assertIsInstance(output_ids, torch.Tensor)
        self.assertEqual(output_ids.dim(), 2)
        self.assertTrue(output_ids.shape[1] > len(input_ids[0]))
        
        # Test decoding
        generated_text = self.tokenizer.decode(output_ids[0].tolist())
        self.assertIsInstance(generated_text, str)
        self.assertGreater(len(generated_text), 0)

    def test_dataset(self):
        """Test dataset loading and processing"""
        train_loader, val_loader = create_dataloaders(
            data_dir='test_data',
            batch_size=self.train_config.batch_size,
            max_seq_len=self.model_config.max_seq_len,
            num_workers=0,  # Use 0 workers for testing
            tokenizer=self.tokenizer
        )
        
        # Test batch loading
        batch = next(iter(train_loader))
        self.assertIn('input_ids', batch)
        self.assertIn('labels', batch)
        self.assertEqual(batch['input_ids'].shape[0], self.train_config.batch_size)
        
        # Test validation loader
        val_batch = next(iter(val_loader))
        self.assertEqual(val_batch['input_ids'].shape[0], self.train_config.batch_size)

    def test_training_step(self):
        """Test one training step"""
        optimizer = self.model.configure_optimizers(
            weight_decay=self.train_config.weight_decay,
            learning_rate=self.train_config.learning_rate,
            betas=(0.9, 0.95),
            device_type=self.device
        )
        
        # Create sample batch
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, 256, (batch_size, seq_len)).to(self.device)
        labels = torch.randint(0, 256, (batch_size, seq_len)).to(self.device)
        
        # Training step
        logits = self.model(input_ids, labels)
        loss = self.model.last_loss
        loss.backward()
        
        # Check gradients
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
        
        optimizer.step()
        optimizer.zero_grad()

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        import shutil
        shutil.rmtree('test_data', ignore_errors=True)

def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2)

if __name__ == '__main__':
    run_tests()
