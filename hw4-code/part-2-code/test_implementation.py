"""
Quick test script to verify the implementation works correctly.
"""

import torch
from load_data import T5Dataset, load_prompting_data

def test_dataset():
    print("Testing T5Dataset...")
    
    # Test training dataset
    train_dataset = T5Dataset('data', 'train')
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Get first sample
    sample = train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Encoder input shape: {sample['encoder_input'].shape}")
    print(f"Decoder input shape: {sample['decoder_input'].shape}")
    print(f"Decoder target shape: {sample['decoder_target'].shape}")
    
    # Test dev dataset
    dev_dataset = T5Dataset('data', 'dev')
    print(f"\nDev dataset size: {len(dev_dataset)}")
    
    # Test test dataset
    test_dataset = T5Dataset('data', 'test')
    print(f"Test dataset size: {len(test_dataset)}")
    test_sample = test_dataset[0]
    print(f"Test sample keys: {test_sample.keys()}")
    
    print("\n✓ Dataset implementation looks good!")

def test_prompting_data():
    print("\nTesting prompting data loading...")
    
    train_x, train_y, dev_x, dev_y, test_x = load_prompting_data('data')
    print(f"Training samples: {len(train_x)} NL, {len(train_y)} SQL")
    print(f"Dev samples: {len(dev_x)} NL, {len(dev_y)} SQL")
    print(f"Test samples: {len(test_x)} NL")
    
    print(f"\nSample NL: {train_x[0]}")
    print(f"Sample SQL: {train_y[0][:100]}...")
    
    print("\n✓ Prompting data loading looks good!")

if __name__ == "__main__":
    test_dataset()
    test_prompting_data()
    print("\n✅ All tests passed!")
