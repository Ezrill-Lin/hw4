import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    parser.add_argument('--freeze_encoder', action='store_true', 
                        help="Freeze encoder layers and only train decoder (faster, may work well for this task)")
    parser.add_argument('--freeze_embeddings', action='store_true',
                        help="Freeze shared embeddings (can help prevent overfitting)")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="Learning rate (1e-4 to 5e-4 recommended for T5 fine-tuning)")
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help="Weight decay for regularization")

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=2,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=20,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=5,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    
    # Optimization flags
    parser.add_argument('--eval_every_n_epochs', type=int, default=1,
                        help="Run full evaluation every N epochs (set >1 to speed up training)")
    parser.add_argument('--skip_record_computation', action='store_true',
                        help="Skip expensive SQL execution during training, only use loss for early stopping")
    parser.add_argument('--save_only_best', action='store_true',
                        help="Only save best model checkpoint, skip saving last model each epoch")

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    experiment_name = 'ft_experiment'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        # Decide whether to run full evaluation this epoch
        should_eval_fully = (epoch % args.eval_every_n_epochs == 0) or (epoch == args.max_n_epochs - 1)
        
        if should_eval_fully:
            eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
                args, model, dev_loader,
                gt_sql_path, model_sql_path,
                gt_record_path, model_record_path
            )
            print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
            print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")
        else:
            # Fast evaluation: only compute loss, skip expensive SQL execution
            print(f"Epoch {epoch}: Running fast evaluation (loss only)...")
            eval_loss = eval_epoch_fast(args, model, dev_loader)
            print(f"Epoch {epoch}: Dev loss: {eval_loss}")
            # Use loss as proxy for F1 (inversely correlated)
            record_f1 = -eval_loss  # Placeholder for comparison

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
            }
            if should_eval_fully:
                result_dict.update({
                    'dev/record_f1' : record_f1,
                    'dev/record_em' : record_em,
                    'dev/sql_em' : sql_em,
                    'dev/error_rate' : error_rate,
                })
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
            # Save best model when improvement occurs
            save_model(checkpoint_dir, model, best=True)
        else:
            epochs_since_improvement += 1

        # Optionally save last model checkpoint
        if not args.save_only_best:
            save_model(checkpoint_dir, model, best=False)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0

    for encoder_input, encoder_mask, labels, _, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            # Count non-padding tokens
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens

def eval_epoch_fast(args, model, dev_loader):
    '''
    Fast evaluation that only computes loss without SQL generation/execution.
    Use this for intermediate epochs to speed up training.
    '''
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for encoder_input, encoder_mask, labels, _, _ in tqdm(dev_loader, desc="Fast eval"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Only compute loss
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    # For generation
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    generated_queries = []
    
    print("Generating SQL queries...")
    with torch.no_grad():
        for encoder_input, encoder_mask, labels, _, _ in tqdm(dev_loader, desc="Evaluating"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Compute loss
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Generate SQL queries with improved parameters
            generated_outputs = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=512,
                num_beams=5,              # Increased beam search
                early_stopping=True,
                no_repeat_ngram_size=3,   # Prevent repetitive patterns
                length_penalty=1.0        # Balance length vs quality
            )
            
            # Decode generated outputs
            for output in generated_outputs:
                decoded = tokenizer.decode(output, skip_special_tokens=True)
                generated_queries.append(decoded)
    
    # Save generated queries and compute records (this is the slow part)
    if args.skip_record_computation:
        # Skip expensive SQL execution, just save queries
        print("Skipping record computation (using --skip_record_computation flag)")
        with open(model_sql_path, 'w') as f:
            for query in generated_queries:
                f.write(f'{query}\n')
        # Return dummy values
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        return avg_loss, 0.0, 0.0, 0.0, 0.0
    else:
        print("Computing database records (this may take 30-60 seconds)...")
        save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    
    # Compute metrics
    print("Computing metrics...")
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    
    # Calculate error rate
    error_count = sum(1 for msg in error_msgs if msg is not None and msg != '')
    error_rate = error_count / len(error_msgs) if len(error_msgs) > 0 else 0
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, record_f1, record_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    
    # For generation
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    generated_queries = []
    
    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            # Generate SQL queries with improved parameters
            outputs = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=512,
                num_beams=5,              # Increased beam search
                early_stopping=True,
                no_repeat_ngram_size=3,   # Prevent repetitive patterns
                length_penalty=1.0        # Balance length vs quality
            )
            
            # Decode generated outputs
            for output in outputs:
                decoded = tokenizer.decode(output, skip_special_tokens=True)
                generated_queries.append(decoded)
    
    # Save generated queries and compute records
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = 'ft_experiment'
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
