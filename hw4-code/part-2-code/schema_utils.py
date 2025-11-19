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
    Uses a tiered approach: always include core tables, then add based on keywords.
    
    Args:
        nl_query: Natural language query
    
    Returns:
        Set of table names that are likely relevant
    """
    relevant_tables = set()
    nl_lower = nl_query.lower()
    
    # Tier 1: Core tables that appear in most queries (always include)
    core_tables = {'flight', 'fare'}
    relevant_tables.update(core_tables)
    
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
    
    # Keyword to table mapping (Tier 2 & 3)
    keyword_to_tables = {
        # Meta/abbreviation queries
        'mean': ['airport', 'airline', 'city', 'aircraft'],
        'abbreviation': ['airport', 'airline', 'aircraft'],
        'code': ['airport', 'airline', 'aircraft'],
        'stand for': ['airport', 'airline', 'aircraft'],
        'what is': ['airport', 'airline', 'city'],
        
        # Flight related (very common)
        'flight': ['airline', 'airport', 'aircraft'],
        'flights': ['airline', 'airport', 'aircraft'],
        'fly': ['airline', 'airport'],
        'airline': ['airline'],
        'airlines': ['airline'],
        'carrier': ['airline'],
        'airport': ['airport', 'airport_service', 'city'],
        'airports': ['airport', 'airport_service', 'city'],
        
        # Location related
        'city': ['city', 'airport', 'airport_service', 'state'],
        'cities': ['city', 'airport', 'airport_service'],
        'from': ['airport', 'city', 'airline'],
        'to': ['airport', 'city', 'airline'],
        'between': ['airport', 'city'],
        'state': ['state', 'city', 'airport'],
        'arrival': ['airport', 'city', 'state'],
        'arrivals': ['airport', 'city', 'state'],
        
        # Cost/pricing related
        'fare': ['fare_basis', 'flight_fare', 'restriction', 'class_of_service'],
        'fares': ['fare_basis', 'flight_fare', 'restriction'],
        'cost': ['fare_basis', 'flight_fare'],
        'price': ['fare_basis', 'flight_fare'],
        'cheap': ['fare_basis', 'flight_fare', 'restriction'],
        'expensive': ['fare_basis', 'flight_fare'],
        'cheapest': ['fare_basis', 'flight_fare', 'restriction'],
        'least': ['fare_basis', 'flight_fare'],
        'lowest': ['fare_basis', 'flight_fare'],
        'minimum': ['fare_basis', 'flight_fare'],
        'maximum': ['fare_basis', 'flight_fare'],
        
        # Aircraft related
        'aircraft': ['aircraft', 'equipment_sequence'],
        'plane': ['aircraft', 'equipment_sequence'],
        'airplane': ['aircraft', 'equipment_sequence'],
        'equipment': ['aircraft', 'equipment_sequence'],
        
        # Service related
        'meal': ['food_service'],
        'food': ['food_service'],
        'breakfast': ['food_service'],
        'lunch': ['food_service'],
        'dinner': ['food_service'],
        'snack': ['food_service'],
        'service': ['class_of_service', 'airport_service', 'ground_service', 'food_service'],
        'class': ['class_of_service', 'fare_basis'],
        'first class': ['class_of_service', 'fare_basis'],
        'business class': ['class_of_service', 'fare_basis'],
        'business': ['class_of_service', 'fare_basis'],
        'economy': ['class_of_service', 'fare_basis'],
        'coach': ['class_of_service', 'fare_basis'],
        'thrift': ['class_of_service', 'fare_basis'],
        
        # Time related
        'time': ['time_interval', 'date_day'],
        'day': ['date_day', 'days'],
        'days': ['date_day', 'days'],
        'date': ['date_day'],
        'week': ['date_day', 'days'],
        'weekday': ['date_day', 'days'],
        'weekend': ['date_day', 'days'],
        'month': ['date_day'],
        'morning': ['time_interval'],
        'afternoon': ['time_interval'],
        'evening': ['time_interval'],
        'night': ['time_interval'],
        'before': ['time_interval', 'date_day'],
        'after': ['time_interval', 'date_day'],
        'early': ['time_interval'],
        'late': ['time_interval'],
        'latest': ['time_interval'],
        'earliest': ['time_interval'],
        
        # Ground transportation
        'ground': ['ground_service', 'airport_service'],
        'transport': ['ground_service'],
        'transportation': ['ground_service'],
        'rental': ['ground_service'],
        'car': ['ground_service'],
        'limousine': ['ground_service'],
        'taxi': ['ground_service'],
        'train': ['ground_service'],
        'bus': ['ground_service'],
        'shuttle': ['ground_service'],
        
        # Restrictions/requirements
        'restriction': ['restriction', 'fare_basis'],
        'restrictions': ['restriction', 'fare_basis'],
        'require': ['restriction', 'fare_basis'],
        'requirement': ['restriction', 'fare_basis'],
        'advance': ['restriction', 'fare_basis'],
        'purchase': ['restriction', 'fare_basis'],
        
        # Stops/connections
        'stop': ['flight_stop', 'flight_leg'],
        'stops': ['flight_stop', 'flight_leg'],
        'nonstop': ['flight_stop', 'flight_leg'],
        'direct': ['flight_stop', 'flight_leg'],
        'connection': ['flight_leg', 'flight_stop'],
        'connecting': ['flight_leg', 'flight_stop'],
        'layover': ['flight_leg', 'flight_stop'],
        
        # Movement/direction
        'departure': ['airport', 'city'],
        'depart': ['airport', 'city'],
        'departing': ['airport', 'city'],
        'arrival': ['airport', 'city'],
        'arrive': ['airport', 'city'],
        'arriving': ['airport', 'city'],
        'leave': ['airport', 'city'],
        'leaving': ['airport', 'city'],
        'return': ['airport', 'city'],
        'returning': ['airport', 'city'],
        'roundtrip': ['airport', 'city'],
        'round trip': ['airport', 'city'],
        'one way': ['airport', 'city'],
    }
    
    for keyword, tables in keyword_to_tables.items():
        if keyword in nl_lower:
            relevant_tables.update(tables)
    
    # Always ensure we have at least the core tables
    if len(relevant_tables) < 2:
        relevant_tables.update(['flight', 'fare', 'airline', 'airport'])
    
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
