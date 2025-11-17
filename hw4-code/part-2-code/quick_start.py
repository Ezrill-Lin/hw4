"""
Quick start training script with recommended hyperparameters.
Run this to verify the training pipeline works before full training.
"""

import subprocess
import sys

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Error running: {description}")
        sys.exit(1)
    print(f"\n✅ Completed: {description}")

def main():
    print("Text-to-SQL Training Quick Start")
    print("=" * 60)
    
    # Test implementation first
    print("\n1. Testing implementation...")
    run_command(
        "python test_implementation.py",
        "Testing data loading implementation"
    )
    
    # Small T5 fine-tuning experiment
    print("\n2. Running small T5 fine-tuning experiment (2 epochs)...")
    run_command(
        "python train_t5.py --finetune --batch_size 16 --test_batch_size 16 "
        "--learning_rate 1e-4 --max_n_epochs 2 --patience_epochs 2 "
        "--scheduler_type cosine --num_warmup_epochs 0 "
        "--experiment_name quick_test",
        "Small T5 fine-tuning test"
    )
    
    print("\n" + "="*60)
    print("✅ Quick start complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Check results in results/ and records/ directories")
    print("2. Run full training with more epochs:")
    print("   python train_t5.py --finetune --batch_size 16 --max_n_epochs 10 ...")
    print("3. Try prompting experiments:")
    print("   python prompting.py --shot 0 --model gemma --experiment_name zero_shot")

if __name__ == "__main__":
    main()
