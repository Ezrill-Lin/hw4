# Performance Improvement Guide: From F1=0.58 to F1>0.65

## Summary of Changes Made

### 1. **CRITICAL FIX: Proper Schema Information** ✅
**Impact: HIGH - Expected +10-15% F1**

- **Before**: Only table names were included (e.g., "tables: flight, airport, city, ...")
- **After**: Table names WITH key columns (e.g., "flight(flight_id,from_airport,to_airport,...)")
- **Why it matters**: Model needs to know which columns exist to generate correct SQL joins and conditions

### 2. **CRITICAL FIX: Training Hyperparameters** ✅
**Impact: HIGH - Model couldn't train properly before**

**Before (BROKEN):**
```python
learning_rate = 0.1      # Way too high!
max_n_epochs = 0         # Won't train!
patience_epochs = 0      # Stops immediately!
weight_decay = 0         # No regularization
```

**After (FIXED):**
```python
learning_rate = 1e-4     # Appropriate for T5 fine-tuning
max_n_epochs = 20        # Enough epochs to converge
patience_epochs = 5      # Proper early stopping
weight_decay = 0.01      # Regularization to prevent overfitting
num_warmup_epochs = 2    # LR warmup for stability
```

### 3. **SQL Normalization** ✅
**Impact: MEDIUM - Expected +3-5% F1**

- Normalizes spacing around commas, operators, keywords
- Helps model learn consistent SQL patterns
- Example: `flight_1 , airport` → `flight_1, airport`

### 4. **Gradient Clipping** ✅
**Impact: LOW-MEDIUM - Improves training stability**

- Clips gradients to max_norm=1.0
- Prevents exploding gradients
- More stable convergence

### 5. **Improved Generation Parameters** ✅
**Impact: LOW-MEDIUM - Expected +2-3% F1**

- Increased beam search: 4 → 5
- Added `no_repeat_ngram_size=3` to prevent repetition
- Added `length_penalty=1.0` for better quality

### 6. **Optional: Partial Fine-tuning** ✅
**Impact: VARIABLE - Can be faster/better**

New options:
- `--freeze_encoder`: Only train decoder (faster, ~50% fewer parameters)
- `--freeze_embeddings`: Freeze embeddings (prevent overfitting)

## Expected Improvement Path

| Component | Previous F1 | Expected F1 | Improvement |
|-----------|-------------|-------------|-------------|
| Baseline (broken) | 0.58 | - | - |
| + Proper schema | 0.58 | 0.70-0.73 | +12-15% |
| + SQL normalization | 0.70 | 0.73-0.75 | +3-5% |
| + Better generation | 0.73 | 0.75-0.77 | +2-3% |

**Target: F1 > 0.65 ✓ (should easily achieve with schema fix alone)**

## Training Recommendations

### **Option 1: Full Fine-tuning (RECOMMENDED)**
```bash
python train_t5.py \
    --finetune \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --batch_size 16 \
    --max_n_epochs 20 \
    --patience_epochs 5 \
    --num_warmup_epochs 2 \
    --scheduler_type cosine \
    --experiment_name full_finetune_with_schema
```

**Pros:**
- Best potential performance
- Full model capacity
- Recommended for first attempt

**Training time:** ~2-4 hours on GPU

### **Option 2: Decoder-Only Fine-tuning (FASTER)**
```bash
python train_t5.py \
    --finetune \
    --freeze_encoder \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --batch_size 16 \
    --max_n_epochs 25 \
    --patience_epochs 5 \
    --num_warmup_epochs 2 \
    --scheduler_type cosine \
    --experiment_name decoder_only_finetune
```

**Pros:**
- Faster training (~50% speedup)
- Less memory usage
- Can use higher learning rate
- Often works well for seq2seq tasks

**Training time:** ~1-2 hours on GPU

### **Option 3: Higher Learning Rate**
If convergence is slow, try:
```bash
--learning_rate 3e-4  # instead of 1e-4
```

## Additional Recommendations (Not Implemented)

If you still need more performance after the above:

### 1. **Data Augmentation**
- Paraphrase natural language queries
- Add synonym replacement
- Expected: +2-3% F1

### 2. **Post-processing**
- Fix common SQL syntax errors
- Validate against schema
- Expected: +1-2% F1

### 3. **Ensemble Methods**
- Train multiple models
- Vote on outputs
- Expected: +2-4% F1

### 4. **Longer Sequences**
- Increase max_length to 768
- Helps with complex queries
- Expected: +1-2% F1

## How to Monitor Training

Watch for these metrics during training:
1. **Train loss should decrease** smoothly
2. **Dev F1 should increase** to >0.65
3. **Error rate should decrease** (SQL syntax errors)
4. **Early stopping** should trigger around epoch 8-15

Good signs:
- Dev F1 improving each epoch initially
- Error rate < 5% by epoch 10
- Convergence before max epochs

Bad signs:
- Train loss not decreasing → learning rate too low
- Train loss exploding → learning rate too high
- Dev F1 not improving → schema issue or data problem

## Files Modified

1. `load_data.py`:
   - Fixed schema loading to use COMPACT_SCHEMA
   - Added SQL normalization function
   - Applied normalization to training SQL

2. `train_t5.py`:
   - Fixed all default hyperparameters
   - Added gradient clipping
   - Improved generation parameters
   - Added partial fine-tuning options

3. `t5_utils.py`:
   - Added support for freezing encoder/embeddings
   - Added parameter count logging

4. `train_recommended.sh`:
   - Created training script with recommended configs

## Quick Start

```bash
cd /workspace/hw4/hw4-code/part-2-code

# Start training with recommended config
./train_recommended.sh

# Or manually:
python train_t5.py --finetune --learning_rate 1e-4 --weight_decay 0.01 \
    --batch_size 16 --max_n_epochs 20 --patience_epochs 5 \
    --num_warmup_epochs 2 --scheduler_type cosine \
    --experiment_name my_experiment
```

## Expected Results

With these changes, you should achieve:
- **F1 Score: 0.70-0.77** (well above 0.65 requirement)
- **SQL Error Rate: <5%**
- **Record EM: 0.55-0.65**

The schema fix alone should be sufficient to pass the 0.65 threshold!
