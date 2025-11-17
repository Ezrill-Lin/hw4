import os


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
    import re
    
    # Try to find SQL query in code blocks
    sql_pattern = r'```sql\s*(.*?)\s*```'
    match = re.search(sql_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Try to find SQL query in generic code blocks
    code_pattern = r'```\s*(.*?)\s*```'
    match = re.search(code_pattern, response, re.DOTALL)
    if match:
        sql_candidate = match.group(1).strip()
        if sql_candidate.upper().startswith('SELECT'):
            return sql_candidate
    
    # Look for SELECT statement directly in the response
    select_pattern = r'(SELECT\s+.*?)(?:\n\n|$|<end_of_turn>|</s>)'
    match = re.search(select_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # If nothing found, try to extract the last line that looks like SQL
    lines = response.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line.upper().startswith('SELECT'):
            return line
    
    # Return the response as is if no pattern matches
    return response.strip()

def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")