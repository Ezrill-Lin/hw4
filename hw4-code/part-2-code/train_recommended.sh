#!/bin/bash
# Recommended training script for achieving F1 > 0.65
# OPTIMIZED FOR H100 - Fast training with minimal gaps between epochs

# Configuration 1: FAST TRAINING - Evaluate every 2 epochs (RECOMMENDED)
# This significantly reduces training time while still monitoring progress
python train_t5.py \
    --finetune \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --batch_size 64 \
    --test_batch_size 64 \
    --max_n_epochs 20 \
    --patience_epochs 5 \
    --num_warmup_epochs 2 \
    --scheduler_type cosine \
    --eval_every_n_epochs 2 \
    --save_only_best \
    --experiment_name fast_training_eval_every_2

# Configuration 2: ULTRA FAST - Loss-only evaluation (if you're in a hurry)
# Only does full evaluation at the end, uses loss for early stopping
# python train_t5.py \
#     --finetune \
#     --learning_rate 1e-4 \
#     --weight_decay 0.01 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --max_n_epochs 20 \
#     --patience_epochs 5 \
#     --num_warmup_epochs 2 \
#     --scheduler_type cosine \
#     --eval_every_n_epochs 5 \
#     --save_only_best \
#     --skip_record_computation \
#     --experiment_name ultra_fast_loss_only

# Configuration 3: Standard Full Evaluation (slower but more monitoring)
# python train_t5.py \
#     --finetune \
#     --learning_rate 1e-4 \
#     --weight_decay 0.01 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --max_n_epochs 20 \
#     --patience_epochs 5 \
#     --num_warmup_epochs 2 \
#     --scheduler_type cosine \
#     --experiment_name full_evaluation_every_epoch
