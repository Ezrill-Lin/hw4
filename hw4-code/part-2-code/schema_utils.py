"""
Utilities for extracting query-specific schema from the full database schema.
"""
import re
from typing import List, Set, Dict

def load_full_schema(schema_path: str) -> Dict[str, str]:
    """
    Load the full schema and parse it into a dictionary.
    
    Returns:
        Dict mapping table_name -> CREATE TABLE statement
    """
    with open(schema_path, 'r') as f:
        schema_text = f.read()
    
    # Parse individual CREATE TABLE statements
    tables = {}
    for match in re.finditer(r'CREATE TABLE (\w+) \((.*?)\);', schema_text, re.DOTALL):
        table_name = match.group(1)
        table_def = match.group(0)
        tables[table_name] = table_def
    
    return tables

def extract_table_names_from_query(nl_query: str) -> Set[str]:
    """
    Extract relevant table names from natural language query using keyword matching heuristics.
    
    Args:
        nl_query: Natural language query
    
    Returns:
        Set of table names that are likely relevant
    """
    relevant_tables = set()
    nl_lower = nl_query.lower()
    
    # All table names - for direct matching
    all_tables = [
        'aircraft', 'airline', 'airport', 'airport_service', 'city',
        'class_of_service', 'date_day', 'days', 'equipment_sequence',
        'fare', 'fare_basis', 'flight', 'flight_fare', 'flight_leg',
        'flight_stop', 'food_service', 'ground_service', 'restriction',
        'state', 'time_interval', 'time_zone'
    ]
    
    # Direct table name matching (if query mentions table name directly)
    for table in all_tables:
        if table.replace('_', ' ') in nl_lower or table in nl_lower:
            relevant_tables.add(table)
    
    # Keyword to table mapping
    keyword_to_tables = {
        'flight': ['flight', 'flight_fare', 'flight_leg', 'flight_stop'],
        'flights': ['flight', 'flight_fare', 'flight_leg', 'flight_stop'],
        'fly': ['flight', 'airline'],
        'airline': ['airline', 'flight'],
        'airlines': ['airline', 'flight'],
        'carrier': ['airline', 'flight'],
        'airport': ['airport', 'airport_service', 'flight'],
        'airports': ['airport', 'airport_service'],
        'city': ['city', 'airport_service'],
        'cities': ['city', 'airport_service'],
        'fare': ['fare', 'fare_basis', 'flight_fare', 'restriction'],
        'fares': ['fare', 'fare_basis', 'flight_fare'],
        'cost': ['fare', 'flight_fare'],
        'price': ['fare', 'flight_fare'],
        'cheap': ['fare', 'flight_fare'],
        'expensive': ['fare', 'flight_fare'],
        'aircraft': ['aircraft', 'equipment_sequence', 'flight'],
        'plane': ['aircraft', 'equipment_sequence', 'flight'],
        'airplane': ['aircraft', 'flight'],
        'meal': ['food_service', 'flight'],
        'food': ['food_service', 'flight'],
        'breakfast': ['food_service', 'flight'],
        'lunch': ['food_service', 'flight'],
        'dinner': ['food_service', 'flight'],
        'snack': ['food_service', 'flight'],
        'service': ['class_of_service', 'airport_service', 'ground_service', 'food_service'],
        'class': ['class_of_service', 'fare_basis', 'fare'],
        'first class': ['class_of_service', 'fare_basis'],
        'business class': ['class_of_service', 'fare_basis'],
        'economy': ['class_of_service', 'fare_basis'],
        'time': ['flight', 'time_interval', 'time_zone', 'date_day'],
        'day': ['date_day', 'days', 'flight'],
        'days': ['date_day', 'days', 'flight'],
        'date': ['date_day', 'flight'],
        'week': ['date_day', 'days'],
        'month': ['date_day'],
        'morning': ['time_interval', 'flight'],
        'afternoon': ['time_interval', 'flight'],
        'evening': ['time_interval', 'flight'],
        'night': ['time_interval', 'flight'],
        'ground': ['ground_service', 'airport_service'],
        'transport': ['ground_service'],
        'transportation': ['ground_service'],
        'restriction': ['restriction', 'fare'],
        'restrictions': ['restriction', 'fare'],
        'state': ['state', 'city', 'airport'],
        'stop': ['flight_stop', 'flight'],
        'stops': ['flight_stop', 'flight'],
        'nonstop': ['flight', 'flight_stop'],
        'connection': ['flight_leg', 'flight'],
        'departure': ['flight', 'airport'],
        'depart': ['flight', 'airport'],
        'arrival': ['flight', 'airport'],
        'arrive': ['flight', 'airport'],
        'leave': ['flight', 'airport'],
        'from': ['flight', 'airport', 'city'],
        'to': ['flight', 'airport', 'city'],
        'between': ['flight', 'airport', 'city'],
    }
    
    for keyword, tables in keyword_to_tables.items():
        if keyword in nl_lower:
            relevant_tables.update(tables)
    
    # If no tables found, include the most common ones
    if not relevant_tables:
        relevant_tables = {'flight', 'airport', 'airline', 'fare'}
    
    return relevant_tables

def get_query_specific_schema(nl_query: str, full_schema_dict: Dict[str, str]) -> str:
    """
    Get a compact schema containing only tables relevant to the query.
    
    Args:
        nl_query: Natural language query
        full_schema_dict: Dictionary of table_name -> CREATE TABLE statement
    
    Returns:
        Compact schema string with only relevant tables
    """
    relevant_tables = extract_table_names_from_query(nl_query)
    
    # Build schema with only relevant tables
    schema_parts = []
    for table_name in sorted(relevant_tables):
        if table_name in full_schema_dict:
            schema_parts.append(full_schema_dict[table_name])
    
    return ' '.join(schema_parts)

def format_input_with_schema(nl_query: str, schema: str, format_type: str = 'detailed') -> str:
    """
    Format the input for the model with schema.
    
    Args:
        nl_query: Natural language query
        schema: Schema text (full or query-specific)
        format_type: 'detailed' (current format) or 'compact' or 'none'
    
    Returns:
        Formatted input string
    """
    if format_type == 'none':
        return f"translate to SQL: {nl_query}"
    elif format_type == 'compact':
        return f"tables: {schema} query: {nl_query}"
    else:  # detailed
        return f"Tables:\n{schema}\n\nQuestion:\n{nl_query}\n\nAnswer:\n"

# Example usage
if __name__ == "__main__":
    # Test the schema extraction
    schema_dict = load_full_schema('data/simplified_schema.txt')
    
    nl = "list all flights from boston to denver"
    
    print("NL Query:", nl)
    print("\nRelevant tables:", extract_table_names_from_query(nl))
    print("\nQuery-specific schema:")
    print(get_query_specific_schema(nl, schema_dict))
    print("\nFormatted input:")
    print(format_input_with_schema(nl, get_query_specific_schema(nl, schema_dict), 'compact'))
