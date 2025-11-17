import os, argparse, random
from tqdm import tqdm

import torch
from transformers import GemmaTokenizerFast, GemmaForCausalLM
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from utils import set_random_seeds, compute_metrics, save_queries_and_records, compute_records
from prompting_utils import read_schema, extract_sql_query, save_logs
from load_data import load_prompting_data

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # you can add mps
MAX_NEW_TOKENS = 512
SCHEMA_PATH = 'data/flight_database.schema'

# Global variables to store training data for few-shot examples
TRAIN_X = None
TRAIN_Y = None


def get_args():
    '''
    Arguments for prompting. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(
        description='Text-to-SQL experiments with prompting.')

    parser.add_argument('-s', '--shot', type=int, default=0,
                        help='Number of examples for k-shot learning (0 for zero-shot)')
    parser.add_argument('-p', '--ptype', type=int, default=0,
                        help='Prompt type')
    parser.add_argument('-m', '--model', type=str, default='gemma',
                        help='Model to use for prompting: gemma (gemma-1.1-2b-it) or codegemma (codegemma-7b-it)')
    parser.add_argument('-q', '--quantization', action='store_true',
                        help='Use a quantized version of the model (e.g. 4bits)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to help reproducibility')
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")
    args = parser.parse_args()
    return args


def create_prompt(sentence, k, schema_text="", train_examples_x=None, train_examples_y=None):
    '''
    Function for creating a prompt for zero or few-shot prompting.

    Add/modify the arguments as needed.

    Inputs:
        * sentence (str): A text string
        * k (int): Number of examples in k-shot prompting
        * schema_text (str): Database schema information
        * train_examples_x (list): Training natural language queries for few-shot
        * train_examples_y (list): Training SQL queries for few-shot
    '''
    # Build the prompt
    prompt = ""
    
    # Add schema information
    if schema_text:
        prompt += f"Tables:\n{schema_text}\n\n"
    
    # Add few-shot examples if k > 0
    if k > 0 and train_examples_x and train_examples_y:
        prompt += "Here are some examples:\n\n"
        
        # Randomly sample k examples
        indices = random.sample(range(len(train_examples_x)), min(k, len(train_examples_x)))
        
        for i, idx in enumerate(indices, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Question: {train_examples_x[idx]}\n"
            prompt += f"Answer: {train_examples_y[idx]}\n\n"
    
    # Add the actual query
    prompt += f"Question:\n{sentence}\n\nAnswer:\n"
    
    return prompt


def exp_kshot(tokenizer, model, inputs, k, train_x=None, train_y=None):
    '''
    k-shot prompting experiments using the provided model and tokenizer. 
    This function generates SQL queries from text prompts and evaluates their accuracy.

    Add/modify the arguments and code as needed.

    Inputs:
        * tokenizer
        * model
        * inputs (List[str]): A list of text strings
        * k (int): Number of examples in k-shot prompting
        * train_x (list): Training natural language queries
        * train_y (list): Training SQL queries
    '''
    raw_outputs = []
    extracted_queries = []
    
    # Read simplified schema
    schema_path = 'data/simplified_schema.txt'
    if os.path.exists(schema_path):
        with open(schema_path, 'r') as f:
            schema_text = f.read().strip()
    else:
        schema_text = ""

    for i, sentence in tqdm(enumerate(inputs)):
        prompt = create_prompt(sentence, k, schema_text, train_x, train_y) # Looking at the prompt may also help

        input_ids = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**input_ids, max_new_tokens=MAX_NEW_TOKENS) # You should set MAX_NEW_TOKENS
        response = tokenizer.decode(outputs[0]) # How does the response look like? You may need to parse it
        raw_outputs.append(response)

        # Extract the SQL query
        extracted_query = extract_sql_query(response)
        extracted_queries.append(extracted_query)
    return raw_outputs, extracted_queries


def eval_outputs(eval_x, eval_y, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    Evaluate the outputs of the model by computing the metrics.

    Add/modify the arguments and code as needed.
    '''
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    
    # Compute error rate
    error_count = sum(1 for msg in model_error_msgs if msg != "")
    error_rate = error_count / len(model_error_msgs) if len(model_error_msgs) > 0 else 0
    
    return sql_em, record_em, record_f1, model_error_msgs, error_rate


def initialize_model_and_tokenizer(model_name, to_quantize=False):
    '''
    Args:
        * model_name (str): Model name ("gemma" or "codegemma").
        * to_quantize (bool): Use a quantized version of the model (e.g. 4bits)
    
    To access to the model on HuggingFace, you need to log in and review the 
    conditions and access the model's content.
    '''
    if model_name == "gemma":
        model_id = "google/gemma-1.1-2b-it"
        tokenizer = GemmaTokenizerFast.from_pretrained(model_id)
        # Native weights exported in bfloat16 precision, but you can use a different precision if needed
        model = GemmaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, 
        ).to(DEVICE)
    elif model_name == "codegemma":
        model_id = "google/codegemma-7b-it"
        tokenizer = GemmaTokenizer.from_pretrained(model_id)
        if to_quantize:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", # 4-bit quantization
            )
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                        torch_dtype=torch.bfloat16,
                                                        config=nf4_config).to(DEVICE)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                        torch_dtype=torch.bfloat16).to(DEVICE)
    return tokenizer, model


def main():
    '''
    Note: this code serves as a basic template for the prompting task. You can but 
    are not required to use this pipeline.
    You can design your own pipeline, and you can also modify the code below.
    '''
    args = get_args()
    shot = args.shot
    ptype = args.ptype
    model_name = args.model
    to_quantize = args.quantization
    experiment_name = args.experiment_name

    set_random_seeds(args.seed)

    data_folder = 'data'
    train_x, train_y, dev_x, dev_y, test_x = load_prompting_data(data_folder)

    # Model and tokenizer
    tokenizer, model = initialize_model_and_tokenizer(model_name, to_quantize)

    for eval_split in ["dev", "test"]:
        eval_x, eval_y = (dev_x, dev_y) if eval_split == "dev" else (test_x, None)

        raw_outputs, extracted_queries = exp_kshot(tokenizer, model, eval_x, shot, train_x, train_y)

        # You can add any post-processing if needed
        # Save queries and compute records
        gt_sql_path = os.path.join(f'data/{eval_split}.sql') if eval_split == "dev" else None
        gt_record_path = os.path.join(f'records/{eval_split}_gt_records.pkl') if eval_split == "dev" else None
        model_sql_path = os.path.join(f'results/{model_name}_{experiment_name}_{eval_split}.sql')
        model_record_path = os.path.join(f'records/{model_name}_{experiment_name}_{eval_split}.pkl')
        
        # Save queries and records
        save_queries_and_records(extracted_queries, model_sql_path, model_record_path)
        
        if eval_split == "dev":
            # Evaluate on dev set
            sql_em, record_em, record_f1, model_error_msgs, error_rate = eval_outputs(
                eval_x, eval_y,
                gt_sql_path, model_sql_path,
                gt_record_path, model_record_path
            )
            print(f"{eval_split} set results: ")
            print(f"Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
            print(f"{eval_split} set results: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

            # Save logs
            log_path = f"logs/{model_name}_{experiment_name}_{eval_split}.txt"
            os.makedirs('logs', exist_ok=True)
            save_logs(log_path, sql_em, record_em, record_f1, model_error_msgs)
        else:
            print(f"Test set inference complete. Results saved to {model_sql_path} and {model_record_path}")


if __name__ == "__main__":
    main()