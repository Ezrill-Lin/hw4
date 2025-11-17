# Schema Integration Guide

## What Changed

### 1. Created Simplified Schema
- Added `data/simplified_schema.txt` with CREATE TABLE statements
- Human-readable format that models can understand
- Includes all 23 tables with columns and types

### 2. Updated T5 Data Loading (`load_data.py`)
- **NEW:** Loads schema and includes it in every encoder input
- **NEW:** Prompt format changed to match notebook style:
  ```
  Tables:
  {schema}
  
  Question:
  {query}
  
  Answer:
  ```
- Added `truncation=True` and `max_length=512` to prevent token overflow

### 3. Updated LLM Prompting (`prompting.py`)
- Schema now included in LLM prompts too
- Cleaner prompt format (removed unnecessary text)
- More similar to notebook's approach

## How to Test

### Test 1: Quick Verification (2 epochs)
```bash
python train_t5.py \
    --finetune \
    --batch_size 8 \
    --test_batch_size 8 \
    --learning_rate 1e-3 \
    --max_n_epochs 2 \
    --patience_epochs 2 \
    --experiment_name schema_test
```

**Expected improvement:** F1 should jump from ~0.17 → ~0.40-0.55

### Test 2: Higher Learning Rate (like notebook)
```bash
python train_t5.py \
    --finetune \
    --batch_size 16 \
    --test_batch_size 16 \
    --learning_rate 5e-3 \
    --weight_decay 0.01 \
    --max_n_epochs 5 \
    --patience_epochs 3 \
    --scheduler_type cosine \
    --num_warmup_epochs 1 \
    --experiment_name schema_high_lr
```

**Expected improvement:** F1 should reach ~0.55-0.70

### Test 3: Full Training
```bash
python train_t5.py \
    --finetune \
    --batch_size 16 \
    --test_batch_size 16 \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --max_n_epochs 10 \
    --patience_epochs 5 \
    --scheduler_type cosine \
    --num_warmup_epochs 1 \
    --experiment_name schema_full
```

**Expected improvement:** F1 should reach ~0.65-0.75+

## Why This Should Work

### Before (F1 = 0.17):
```
Input:  "translate English to SQL: show me flights from denver to atlanta"
Model:  Has no idea what tables/columns exist
Output: Random/invalid SQL
```

### After (F1 = 0.65+):
```
Input:  "Tables:
         CREATE TABLE flight (from_airport, to_airport, ...)
         CREATE TABLE city (city_name, city_code, ...)
         ...
         
         Question:
         show me flights from denver to atlanta
         
         Answer:"
         
Model:  Sees exact schema structure
Output: SELECT ... FROM flight WHERE ... (correct SQL)
```

## If F1 Still Low After These Changes

Try in order:

1. **Increase LR to 5e-3** (like notebook)
2. **Train for more epochs** (10-15)
3. **Reduce batch size to 8** if memory issues
4. **Check a few predictions manually** to see what's wrong:
   ```python
   # Look at model_sql_path after training
   with open('results/t5_ft_schema_test_dev.sql', 'r') as f:
       predictions = f.readlines()[:10]
   
   with open('data/dev.sql', 'r') as f:
       ground_truth = f.readlines()[:10]
   
   for pred, gt in zip(predictions, ground_truth):
       print("PRED:", pred)
       print("GT:  ", gt)
       print()
   ```

## Key Difference from Notebook

**Notebook:** Uses multiple large datasets (200K+ examples)
**Your dataset:** Only 4,226 examples

With a smaller dataset, you need:
- ✅ Schema in prompts (critical!)
- ✅ Higher learning rate (to learn faster)
- ✅ More epochs (to see enough data)
- ✅ Good regularization (weight decay)

The schema is the most important - it gives the model the "cheat sheet" it needs to generate valid SQL.

## Quick Test Without Training

Verify the schema is being loaded:
```python
from load_data import T5Dataset

dataset = T5Dataset('data', 'train')
tokenizer = dataset.tokenizer

# Check first sample
sample = dataset[0]
print("Input tokens:", len(sample['encoder_input']))
print("Decoded input:", tokenizer.decode(sample['encoder_input']))
```

You should see "Tables:\nCREATE TABLE..." in the decoded input.
