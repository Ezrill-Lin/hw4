import os
import argparse
from tqdm import tqdm

import torch

from t5_utils import load_model_from_checkpoint
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_args():
    parser = argparse.ArgumentParser(description='Test best T5 model')
    
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    parser.add_argument('--experiment_name', type=str, required=True,
                        help="Name of the experiment to load")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    
    args = parser.parse_args()
    return args

def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''Evaluation with generation and metrics'''
    model.eval()
    generated_queries = []
    
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    with torch.no_grad():
        for batch in tqdm(dev_loader):
            if len(batch) == 5:  # dev loader
                encoder_input, encoder_mask, _, _, _ = batch
            else:  # test loader
                encoder_input, encoder_mask, _ = batch
                
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            # Generate SQL queries
            outputs = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode generated queries
            for output in outputs:
                query = tokenizer.decode(output, skip_special_tokens=True)
                generated_queries.append(query)
    
    # Save generated queries and compute metrics
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    
    # Compute error rate
    error_count = sum(1 for msg in model_error_msgs if msg != "")
    error_rate = error_count / len(model_error_msgs) if len(model_error_msgs) > 0 else 0
    
    return record_f1, record_em, sql_em, error_rate

def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''Inference on test set'''
    model.eval()
    generated_queries = []
    
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    with torch.no_grad():
        for encoder_input, encoder_mask, initial_decoder_input in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            # Generate SQL queries
            outputs = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode generated queries
            for output in outputs:
                query = tokenizer.decode(output, skip_special_tokens=True)
                generated_queries.append(query)
    
    # Save generated queries and records
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)

def main():
    args = get_args()
    
    # Set checkpoint directory
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    args.checkpoint_dir = checkpoint_dir
    
    # Load data
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    
    # Load best model
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set evaluation
    experiment_name = args.experiment_name
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    
    print("Evaluating on dev set...")
    dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    print(f"\nDev set results:")
    print(f"  Record F1: {dev_record_f1}")
    print(f"  Record EM: {dev_record_em}")
    print(f"  SQL EM: {dev_sql_em}")
    print(f"  Error rate: {dev_error_rate*100:.2f}%")
    
    # Test set inference
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    
    print("\nGenerating predictions on test set...")
    test_inference(args, model, test_loader, model_sql_path, model_record_path)
    print(f"Test predictions saved to {model_sql_path}")

if __name__ == "__main__":
    main()
