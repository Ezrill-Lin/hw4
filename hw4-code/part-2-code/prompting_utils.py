import os
import re


def read_schema(schema_path):
    '''
    Read the .schema file
    '''
    with open(schema_path, 'r') as f:
        schema_content = f.read()
    return schema_content

def extract_sql_query(response):
    '''
    Extract the SQL query from the model's response
    '''
    # Try to extract SQL query from various formats
    
    # Pattern 1: Look for SQL between code blocks (```sql ... ```)
    sql_block_pattern = r'```sql\s*(.*?)\s*```'
    match = re.search(sql_block_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern 2: Look for SQL between generic code blocks (``` ... ```)
    code_block_pattern = r'```\s*(.*?)\s*```'
    match = re.search(code_block_pattern, response, re.DOTALL)
    if match:
        content = match.group(1).strip()
        # Check if it looks like SQL
        if content.upper().startswith('SELECT'):
            return content
    
    # Pattern 3: Look for SELECT statement directly in the response
    # Extract from SELECT to semicolon or end of string
    select_pattern = r'(SELECT\s+.*?)(?:;|\n\n|$)'
    match = re.search(select_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern 4: If response starts with SELECT, take everything
    if response.strip().upper().startswith('SELECT'):
        # Remove trailing semicolon if present
        return response.strip().rstrip(';').strip()
    
    # Pattern 5: Look for common LLM prefixes and extract what comes after
    prefixes = [
        r'(?:SQL query|Query|Answer|Solution|Result):\s*',
        r'(?:Here is the|The) (?:SQL query|query):\s*',
    ]
    
    for prefix in prefixes:
        pattern = prefix + r'(SELECT\s+.*?)(?:;|\n\n|$)'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If nothing found, return the response as is (might be just the SQL query)
    # Remove common wrapper text and clean up
    cleaned = response.strip()
    
    # Remove common instruction-like text at the beginning
    cleaned = re.sub(r'^(?:Here is|The answer is|The SQL query is|Query):\s*', '', cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip().rstrip(';').strip()

def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")