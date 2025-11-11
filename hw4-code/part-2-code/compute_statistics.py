"""
Script to compute data statistics for T5 text-to-SQL task.
This computes statistics both before and after preprocessing.
"""

import os
from transformers import T5TokenizerFast
from load_data import load_lines

def compute_statistics():
    """
    Compute statistics for the dataset before and after tokenization.
    """
    # Initialize T5 tokenizer
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    data_folder = 'data'
    
    # Load raw data
    train_nl = load_lines(os.path.join(data_folder, 'train.nl'))
    train_sql = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_nl = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_sql = load_lines(os.path.join(data_folder, 'dev.sql'))
    
    print("=" * 80)
    print("TABLE 1: BEFORE PREPROCESSING (Raw Text)")
    print("=" * 80)
    
    # Before preprocessing - raw text statistics
    print("\n1. Number of Examples:")
    print(f"   Train: {len(train_nl)}")
    print(f"   Dev: {len(dev_nl)}")
    
    # Mean sentence length (in words/characters)
    train_nl_lengths = [len(s.split()) for s in train_nl]
    dev_nl_lengths = [len(s.split()) for s in dev_nl]
    
    print("\n2. Mean Sentence Length (words):")
    print(f"   Train: {sum(train_nl_lengths) / len(train_nl_lengths):.2f}")
    print(f"   Dev: {sum(dev_nl_lengths) / len(dev_nl_lengths):.2f}")
    
    # Mean SQL query length (in words)
    train_sql_lengths = [len(s.split()) for s in train_sql]
    dev_sql_lengths = [len(s.split()) for s in dev_sql]
    
    print("\n3. Mean SQL Query Length (words):")
    print(f"   Train: {sum(train_sql_lengths) / len(train_sql_lengths):.2f}")
    print(f"   Dev: {sum(dev_sql_lengths) / len(dev_sql_lengths):.2f}")
    
    # Vocabulary size (unique words) - before tokenization
    train_nl_vocab = set()
    for s in train_nl:
        train_nl_vocab.update(s.lower().split())
    
    dev_nl_vocab = set()
    for s in dev_nl:
        dev_nl_vocab.update(s.lower().split())
    
    print("\n4. Vocabulary Size - Natural Language (unique words):")
    print(f"   Train: {len(train_nl_vocab)}")
    print(f"   Dev: {len(dev_nl_vocab)}")
    
    # Vocabulary size for SQL
    train_sql_vocab = set()
    for s in train_sql:
        train_sql_vocab.update(s.split())
    
    dev_sql_vocab = set()
    for s in dev_sql:
        dev_sql_vocab.update(s.split())
    
    print("\n5. Vocabulary Size - SQL (unique tokens):")
    print(f"   Train: {len(train_sql_vocab)}")
    print(f"   Dev: {len(dev_sql_vocab)}")
    
    print("\n" + "=" * 80)
    print("TABLE 2: AFTER PREPROCESSING (Tokenized with T5 Tokenizer)")
    print("=" * 80)
    
    # After preprocessing - tokenized statistics
    print("\n1. Number of Examples:")
    print(f"   Train: {len(train_nl)}")
    print(f"   Dev: {len(dev_nl)}")
    
    # Tokenize and compute lengths
    train_nl_token_lengths = []
    for s in train_nl:
        tokens = tokenizer(s, add_special_tokens=True)['input_ids']
        train_nl_token_lengths.append(len(tokens))
    
    dev_nl_token_lengths = []
    for s in dev_nl:
        tokens = tokenizer(s, add_special_tokens=True)['input_ids']
        dev_nl_token_lengths.append(len(tokens))
    
    print("\n2. Mean Sentence Length (tokens):")
    print(f"   Train: {sum(train_nl_token_lengths) / len(train_nl_token_lengths):.2f}")
    print(f"   Dev: {sum(dev_nl_token_lengths) / len(dev_nl_token_lengths):.2f}")
    
    # SQL token lengths
    train_sql_token_lengths = []
    for s in train_sql:
        tokens = tokenizer(s, add_special_tokens=True)['input_ids']
        train_sql_token_lengths.append(len(tokens))
    
    dev_sql_token_lengths = []
    for s in dev_sql:
        tokens = tokenizer(s, add_special_tokens=True)['input_ids']
        dev_sql_token_lengths.append(len(tokens))
    
    print("\n3. Mean SQL Query Length (tokens):")
    print(f"   Train: {sum(train_sql_token_lengths) / len(train_sql_token_lengths):.2f}")
    print(f"   Dev: {sum(dev_sql_token_lengths) / len(dev_sql_token_lengths):.2f}")
    
    # Vocabulary size after tokenization (unique token IDs)
    train_nl_token_vocab = set()
    for s in train_nl:
        tokens = tokenizer(s, add_special_tokens=True)['input_ids']
        train_nl_token_vocab.update(tokens)
    
    dev_nl_token_vocab = set()
    for s in dev_nl:
        tokens = tokenizer(s, add_special_tokens=True)['input_ids']
        dev_nl_token_vocab.update(tokens)
    
    print("\n4. Vocabulary Size - Natural Language (unique token IDs):")
    print(f"   Train: {len(train_nl_token_vocab)}")
    print(f"   Dev: {len(dev_nl_token_vocab)}")
    
    # SQL token vocabulary
    train_sql_token_vocab = set()
    for s in train_sql:
        tokens = tokenizer(s, add_special_tokens=True)['input_ids']
        train_sql_token_vocab.update(tokens)
    
    dev_sql_token_vocab = set()
    for s in dev_sql:
        tokens = tokenizer(s, add_special_tokens=True)['input_ids']
        dev_sql_token_vocab.update(tokens)
    
    print("\n5. Vocabulary Size - SQL (unique token IDs):")
    print(f"   Train: {len(train_sql_token_vocab)}")
    print(f"   Dev: {len(dev_sql_token_vocab)}")
    
    print("\n" + "=" * 80)
    print("ADDITIONAL STATISTICS")
    print("=" * 80)
    
    print(f"\nT5 Tokenizer Vocabulary Size: {tokenizer.vocab_size}")
    print(f"\nMax sentence length (tokens):")
    print(f"   Train NL: {max(train_nl_token_lengths)}")
    print(f"   Dev NL: {max(dev_nl_token_lengths)}")
    print(f"   Train SQL: {max(train_sql_token_lengths)}")
    print(f"   Dev SQL: {max(dev_sql_token_lengths)}")
    
    print("\n" + "=" * 80)
    print("\nFormatted for LaTeX Table:")
    print("=" * 80)
    
    print("\nTable 1 (Before Preprocessing):")
    print("Statistics & Train & Dev \\\\")
    print("\\hline")
    print(f"Number of examples & {len(train_nl)} & {len(dev_nl)} \\\\")
    print(f"Mean sentence length & {sum(train_nl_lengths) / len(train_nl_lengths):.2f} & {sum(dev_nl_lengths) / len(dev_nl_lengths):.2f} \\\\")
    print(f"Mean SQL query length & {sum(train_sql_lengths) / len(train_sql_lengths):.2f} & {sum(dev_sql_lengths) / len(dev_sql_lengths):.2f} \\\\")
    print(f"Vocabulary size (NL) & {len(train_nl_vocab)} & {len(dev_nl_vocab)} \\\\")
    print(f"Vocabulary size (SQL) & {len(train_sql_vocab)} & {len(dev_sql_vocab)} \\\\")
    
    print("\nTable 2 (After Preprocessing with T5 Tokenizer):")
    print("Statistics & Train & Dev \\\\")
    print("\\hline")
    print(f"Number of examples & {len(train_nl)} & {len(dev_nl)} \\\\")
    print(f"Mean sentence length & {sum(train_nl_token_lengths) / len(train_nl_token_lengths):.2f} & {sum(dev_nl_token_lengths) / len(dev_nl_token_lengths):.2f} \\\\")
    print(f"Mean SQL query length & {sum(train_sql_token_lengths) / len(train_sql_token_lengths):.2f} & {sum(dev_sql_token_lengths) / len(dev_sql_token_lengths):.2f} \\\\")
    print(f"Vocabulary size (NL) & {len(train_nl_token_vocab)} & {len(dev_nl_token_vocab)} \\\\")
    print(f"Vocabulary size (SQL) & {len(train_sql_token_vocab)} & {len(dev_sql_token_vocab)} \\\\")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    compute_statistics()
