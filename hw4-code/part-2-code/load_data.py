import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
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
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.encoder_inputs, self.decoder_inputs, self.decoder_targets = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # Load database schema
        schema_path = os.path.join(data_folder, 'simplified_schema.txt')
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                schema = f.read().strip()
        
        # Load natural language queries
        nl_path = os.path.join(data_folder, f'{split}.nl')
        with open(nl_path, 'r') as f:
            nl_queries = [line.strip() for line in f.readlines()]
        
        # Tokenize encoder inputs (natural language queries)
        encoder_inputs = []
        for query in nl_queries:
            # Build input with schema
            input_text = f"Tables:\n{schema}\n\nQuestion:\n{query}\n\nAnswer:\n"
            encoder_input = tokenizer(input_text, return_tensors='pt', add_special_tokens=True)
            encoder_inputs.append(encoder_input['input_ids'].squeeze(0))
        
        # For test set, we don't have SQL targets
        if split == 'test':
            return encoder_inputs, None, None
        
        # Load SQL queries
        sql_path = os.path.join(data_folder, f'{split}.sql')
        with open(sql_path, 'r') as f:
            sql_queries = [line.strip() for line in f.readlines()]
        
        # Tokenize decoder inputs and targets
        decoder_inputs = []
        decoder_targets = []
        for sql in sql_queries:
            # Tokenize the SQL query (no truncation - SQL should fit)
            sql_tokens = tokenizer(sql, return_tensors='pt', add_special_tokens=True)
            sql_ids = sql_tokens['input_ids'].squeeze(0)
            
            # Decoder input: start with BOS token (using pad token as decoder start)
            decoder_input = torch.cat([torch.tensor([tokenizer.pad_token_id]), sql_ids[:-1]])
            decoder_inputs.append(decoder_input)
            
            # Decoder target: the SQL tokens (shifted by one position)
            decoder_targets.append(sql_ids)
        
        return encoder_inputs, decoder_inputs, decoder_targets
    
    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        if self.split == 'test':
            return {
                'encoder_input': self.encoder_inputs[idx],
                'initial_decoder_input': torch.tensor([self.tokenizer.pad_token_id])
            }
        else:
            return {
                'encoder_input': self.encoder_inputs[idx],
                'decoder_input': self.decoder_inputs[idx],
                'decoder_target': self.decoder_targets[idx],
                'initial_decoder_input': torch.tensor([self.tokenizer.pad_token_id])
            }

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
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_inputs = [item['encoder_input'] for item in batch]
    decoder_inputs = [item['decoder_input'] for item in batch]
    decoder_targets = [item['decoder_target'] for item in batch]
    initial_decoder_inputs = [item['initial_decoder_input'] for item in batch]
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Pad decoder inputs and targets
    decoder_input_ids = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_target_ids = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    
    # Initial decoder inputs (batch of single tokens)
    initial_decoder_input_ids = torch.stack(initial_decoder_inputs).squeeze(-1)
    if len(initial_decoder_input_ids.shape) == 1:
        initial_decoder_input_ids = initial_decoder_input_ids.unsqueeze(-1)
    
    return encoder_ids, encoder_mask, decoder_input_ids, decoder_target_ids, initial_decoder_input_ids

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_inputs = [item['encoder_input'] for item in batch]
    initial_decoder_inputs = [item['initial_decoder_input'] for item in batch]
    
    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # Initial decoder inputs
    initial_decoder_input_ids = torch.stack(initial_decoder_inputs).squeeze(-1)
    if len(initial_decoder_input_ids.shape) == 1:
        initial_decoder_input_ids = initial_decoder_input_ids.unsqueeze(-1)
    
    return encoder_ids, encoder_mask, initial_decoder_input_ids

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # Load training data
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    
    # Load dev data
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    
    # Load test data (no labels)
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    
    return train_x, train_y, dev_x, dev_y, test_x