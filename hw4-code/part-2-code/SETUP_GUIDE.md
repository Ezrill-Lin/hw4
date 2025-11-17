# Setup and Training Guide

## Environment Setup

1. **Create virtual environment:**
   ```bash
   conda create -n hw4-part-2-nlp python=3.10
   conda activate hw4-part-2-nlp
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data:**
   ```python
   python -c "import nltk; nltk.download('punkt')"
   ```

4. **Create ground truth records (optional but recommended):**
   ```bash
   python create_ground_truth.py
   ```

## Verify Installation

Run the test script to ensure everything is set up correctly:
```bash
python test_implementation.py
```

Expected output:
- Training dataset size: 4226
- Dev dataset size: 467
- Test dataset size: 432

## Training Options

### Option 1: T5 Fine-tuning (Recommended)

**Quick test (2 epochs):**
```bash
python train_t5.py \
    --finetune \
    --batch_size 8 \
    --test_batch_size 8 \
    --learning_rate 1e-4 \
    --max_n_epochs 2 \
    --patience_epochs 2 \
    --scheduler_type cosine \
    --experiment_name quick_test
```

**Full training:**
```bash
python train_t5.py \
    --finetune \
    --batch_size 16 \
    --test_batch_size 16 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_n_epochs 10 \
    --patience_epochs 3 \
    --scheduler_type cosine \
    --num_warmup_epochs 1 \
    --experiment_name ft_final \
    --use_wandb
```

**Expected time:** ~30-60 min per epoch on GPU, ~2-3 hours on CPU

### Option 2: T5 From Scratch

```bash
python train_t5.py \
    --batch_size 16 \
    --test_batch_size 16 \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --max_n_epochs 20 \
    --patience_epochs 5 \
    --scheduler_type cosine \
    --num_warmup_epochs 2 \
    --experiment_name scr_final \
    --use_wandb
```

**Expected time:** Longer training required (20+ epochs)

### Option 3: LLM Prompting

**Zero-shot with Gemma:**
```bash
python prompting.py \
    --shot 0 \
    --model gemma \
    --experiment_name gemma_zero
```

**Few-shot with Gemma (5 examples):**
```bash
python prompting.py \
    --shot 5 \
    --model gemma \
    --experiment_name gemma_five
```

**Zero-shot with CodeGemma (quantized):**
```bash
python prompting.py \
    --shot 0 \
    --model codegemma \
    --quantization \
    --experiment_name codegemma_zero
```

**Expected time:** 
- Gemma: ~15-30 min for dev+test
- CodeGemma: ~30-60 min for dev+test

## Monitoring Training

### Without WandB
Monitor the console output:
- Training loss per epoch
- Dev set: Record F1, Record EM, SQL EM, Error rate
- Best model saved when F1 improves

### With WandB
1. Create account at https://wandb.ai
2. Login: `wandb login`
3. Add `--use_wandb` flag to training command
4. View real-time metrics at wandb.ai

## Output Files

### During Training
- `checkpoints/{experiment_name}/best_model.pt` - Best model (highest F1)
- `checkpoints/{experiment_name}/last_model.pt` - Most recent model

### After Training
- `results/t5_{ft|scr}_{experiment}_dev.sql` - Dev predictions
- `results/t5_{ft|scr}_{experiment}_test.sql` - Test predictions
- `records/t5_{ft|scr}_{experiment}_dev.pkl` - Dev records
- `records/t5_{ft|scr}_{experiment}_test.pkl` - Test records

### For Prompting
- `results/{model}_{experiment}_{dev|test}.sql` - SQL predictions
- `records/{model}_{experiment}_{dev|test}.pkl` - Database records
- `logs/{model}_{experiment}_dev.txt` - Evaluation metrics

## Evaluation

To evaluate saved predictions:
```bash
python evaluate.py \
    --predicted_sql results/t5_ft_experiment_dev.sql \
    --predicted_records records/t5_ft_experiment_dev.pkl \
    --development_sql data/dev.sql \
    --development_records records/dev_gt_records.pkl
```

## Submission Files

For final submission, rename your best test files to:
- T5 fine-tuned: `t5_ft_test.sql` and `t5_ft_test.pkl`
- T5 from scratch: `t5_scr_test.sql` and `t5_scr_test.pkl`
- LLM (choose one): `gemma_test.sql` and `gemma_test.pkl`

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch_size` (try 8 or 4)
- Reduce `--test_batch_size`
- Use gradient accumulation (modify code)

### Training Too Slow
- Use smaller dev set for faster evaluation (modify code)
- Reduce beam size in generation (modify code)
- Use greedy decoding instead of beam search

### Poor Results
- Try different learning rates (1e-4, 5e-5, 1e-5)
- Increase training epochs
- Adjust warmup epochs
- Experiment with different prompt formats (for LLM)

## Recommended Workflow

1. **Test implementation:** `python test_implementation.py`
2. **Create ground truth:** `python create_ground_truth.py`
3. **Quick test:** Run 2-epoch experiment to verify training works
4. **Full T5 fine-tuning:** Best performance expected here
5. **LLM experiments:** Try zero-shot and few-shot
6. **Compare results:** Use dev set metrics to choose best model
7. **Submit:** Rename test files and submit

## Tips for Better Performance

### T5 Training
- Learning rate 1e-4 works well for fine-tuning
- Cosine scheduler with 1 warmup epoch recommended
- Monitor dev F1, not loss
- Early stopping at 3 epochs patience prevents overfitting

### LLM Prompting
- More shots (3-5) usually better than zero-shot
- CodeGemma may perform better than Gemma on SQL
- Quantization makes CodeGemma feasible on smaller GPUs
- Schema information helps but keep prompts concise

### General
- Dev set F1 is the primary metric
- SQL EM is often lower than Record F1 (expected)
- Some SQL errors are normal (~5-10%)
- Beam search (4 beams) better than greedy decoding

## Expected Performance Ranges

Based on the dataset:
- **T5 Fine-tuned:** Record F1 ~0.65-0.80+
- **T5 From Scratch:** Record F1 ~0.40-0.60
- **LLM Zero-shot:** Record F1 ~0.30-0.50
- **LLM Few-shot:** Record F1 ~0.45-0.65

*Actual results may vary based on hyperparameters and compute resources.*
