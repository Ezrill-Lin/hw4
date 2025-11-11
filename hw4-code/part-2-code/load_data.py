import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle
import json

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split, use_schema=True):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.use_schema = use_schema
        self.data_folder = data_folder
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def load_schema_info(self, schema_path='data/flight_database.schema'):
        """
        Load and parse the database schema file provided in the assignment.
        
        Given T5-small's 512 token limit, we use a compact schema representation:
        just table names without full column lists. The model can still learn
        table relationships and common column patterns from the training data.
        
        This is a practical trade-off between providing schema context and staying
        within token limits.
        """
        try:
            with open(schema_path, 'r') as f:
                schema_data = json.load(f)
            
            # Extract just table names - most compact representation
            tables = []
            if 'ents' in schema_data:
                tables = list(schema_data['ents'].keys())
            
            # Create compact schema: just list table names
            schema_str = "tables: " + ", ".join(tables)
            return schema_str
        except Exception as e:
            print(f"Warning: Could not load schema file: {e}")
            # Fallback to basic schema if file can't be loaded
            return "tables: flight, airport, city, state, airline, fare"

    def process_data(self, data_folder, split, tokenizer):
        """
        Process the data for T5 model training and inference.
        
        Data processing includes:
        1. Loading natural language queries from .nl files
        2. Loading SQL queries from .sql files (for train/dev)
        3. Optionally augmenting input with schema information from provided schema file
        4. Tokenizing inputs and outputs using T5 tokenizer
        """
        # Load natural language queries
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_queries = load_lines(nl_path)
        
        # Load schema information if enabled
        schema_str = None
        if self.use_schema:
            schema_str = self.load_schema_info()
        
        processed_data = []
        
        # For test set, we don't have SQL queries
        if split == 'test':
            for nl in nl_queries:
                # Construct input text with optional schema information
                if schema_str:
                    # Add schema context to help model generate correct SQL
                    input_text = f"translate to SQL: {nl} | schema: {schema_str}"
                else:
                    input_text = f"translate to SQL: {nl}"
                    
                # Tokenize the natural language input
                encoder_inputs = tokenizer(input_text, return_tensors='pt', add_special_tokens=True)
                encoder_input_ids = encoder_inputs['input_ids'].squeeze(0)
                
                processed_data.append({
                    'encoder_input_ids': encoder_input_ids
                })
        else:
            # For train and dev sets, we have both NL and SQL
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_queries = load_lines(sql_path)
            
            assert len(nl_queries) == len(sql_queries), f"Mismatch in data sizes for {split}"
            
            for nl, sql in zip(nl_queries, sql_queries):
                # Construct input text with optional schema information
                if schema_str:
                    # Add schema context to help model generate correct SQL
                    input_text = f"translate to SQL: {nl} | schema: {schema_str}"
                else:
                    input_text = f"translate to SQL: {nl}"
                    
                # Tokenize the natural language input
                encoder_inputs = tokenizer(input_text, return_tensors='pt', add_special_tokens=True)
                encoder_input_ids = encoder_inputs['input_ids'].squeeze(0)
                
                # Tokenize the SQL output (T5 handles decoder start token automatically)
                decoder_inputs = tokenizer(sql, return_tensors='pt', add_special_tokens=True)
                decoder_input_ids = decoder_inputs['input_ids'].squeeze(0)
                
                processed_data.append({
                    'encoder_input_ids': encoder_input_ids,
                    'decoder_input_ids': decoder_input_ids
                })
        
        return processed_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder (labels for teacher forcing)
        * decoder_targets: Not used with T5's standard training, can return None or same as decoder_inputs
        * initial_decoder_inputs: Not used, can return None
    '''
    # Extract encoder and decoder sequences
    encoder_input_ids = [item['encoder_input_ids'] for item in batch]
    decoder_input_ids = [item['decoder_input_ids'] for item in batch]
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_input_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Pad decoder inputs (these will be used as labels in T5)
    decoder_inputs = pad_sequence(decoder_input_ids, batch_first=True, padding_value=PAD_IDX)
    
    # For T5, we use labels directly - the model handles shifting internally
    # Set padding tokens to -100 so they're ignored in loss calculation
    decoder_labels = decoder_inputs.clone()
    decoder_labels[decoder_labels == PAD_IDX] = -100
    
    return encoder_ids, encoder_mask, decoder_labels, None, None

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: Not needed for T5, return None
    '''
    # Extract encoder sequences
    encoder_input_ids = [item['encoder_input_ids'] for item in batch]
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_input_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    return encoder_ids, encoder_mask, None

def get_dataloader(batch_size, split, use_schema=True):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split, use_schema=use_schema)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size, use_schema=True):
    train_loader = get_dataloader(batch_size, "train", use_schema=use_schema)
    dev_loader = get_dataloader(test_batch_size, "dev", use_schema=use_schema)
    test_loader = get_dataloader(test_batch_size, "test", use_schema=use_schema)
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # Load training data (for few-shot examples)
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    
    # Load development data
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    
    # Load test data (no labels)
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    
    return train_x, train_y, dev_x, dev_y, test_x