#!/usr/bin/env python
"""
RunPod-specific training script that handles CUDA initialization issues.
This script explicitly manages CUDA device selection before importing PyTorch.
"""
import os
import sys

# CRITICAL: Set CUDA device BEFORE importing torch
# This prevents context conflicts on RunPod
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Now safe to import torch
import torch
import gc

# Force clear any existing CUDA cache
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Explicitly set device
    torch.cuda.set_device(0)
    print(f"CUDA initialized: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
else:
    print("WARNING: CUDA not available, training on CPU")

# Now import and run the main training script
from train_t5 import main

if __name__ == "__main__":
    main()
