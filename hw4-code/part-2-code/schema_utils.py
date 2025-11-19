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
    Uses a precision-focused approach: start with minimal tables and add only when strong signals present.
    
    Args:
        nl_query: Natural language query
    
    Returns:
        Set of table names that are likely relevant
    """
    relevant_tables = set()
    nl_lower = nl_query.lower()
    
    # Start with flight table (appears in 74% of all queries)
    relevant_tables.add('flight')
    
    # Strong signals for specific tables (high precision keywords)
    strong_signals = {
        'fare': ['fare', 'fares', 'cost', 'price', 'cheap', 'expensive', 'cheapest', 'least expensive', 'how much'],
        'ground_service': ['ground transportation', 'ground', 'limousine', 'rental', 'car rental', 'taxi', 'train'],
        'airline': ['airline', 'airlines', 'carrier', 'ff', 'co', 'dl', 'ua', 'aa', 'wn', 'canadian airlines', 'lufthansa', 'twa', 'united'],
        'aircraft': ['aircraft', 'type of aircraft', 'what aircraft', 'what type', 'plane type', 'what kind of plane', 'capacity', 'seats', 'f28', 'dc10', 'boeing', '727', '737', '747', '734'],
        'restriction': ['restriction', 'restrictions', 'advance purchase', 'requirement'],
        'class_of_service': ['first class', 'business class', 'coach', 'thrift', 'economy class', 'classes of service', 'what are the different classes'],
        'food_service': ['meal', 'breakfast', 'dinner', 'lunch', 'snack', 'food service'],
        'airport': ['what does', 'what is', 'abbreviation', 'airport code', 'mean', 'where does'],
        'state': ['state', 'arrivals', 'departures', 'leaving on'],
        'city': ['city', 'cities'],
        'airport_service': ['airport service', 'how far from the airport'],
        'fare_basis': ['fare code', 'fare basis', 'code y', 'what does code', 'yn code', 'stand for'],
        'flight_stop': ['stop', 'stops', 'stopover'],
        'flight_leg': ['connection', 'connecting'],
        'date_day': ['daily', 'weekday', 'weekend'],
        'days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'what is sa'],
        'time_interval': ['morning', 'afternoon', 'evening', 'night', 'before', 'after'],
    }
    
    # Check for strong signals
    for table, keywords in strong_signals.items():
        for keyword in keywords:
            if keyword in nl_lower:
                relevant_tables.add(table)
                break
    
    # Special case: if asking about fares, also include fare_basis and flight_fare
    if 'fare' in relevant_tables:
        relevant_tables.update(['fare_basis', 'flight_fare'])
    
    # Special case: if ground_service mentioned, likely need airport too
    if 'ground_service' in relevant_tables:
        relevant_tables.add('airport')
    
    # Special case: if asking about time, probably need dates
    if 'time_interval' in relevant_tables or any(day in nl_lower for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
        relevant_tables.add('date_day')
    
    # Special case: location queries may need airport/city linkage
    if 'airport' in relevant_tables or any(word in nl_lower for word in ['from', 'to', 'between', 'where does']):
        if not any(t in relevant_tables for t in ['ground_service', 'fare']):
            # Pure flight query with locations
            if 'airline' in relevant_tables and 'where does' in nl_lower:
                # "where does <airline> fly" needs airport
                relevant_tables.add('airport')
            elif 'airline' not in relevant_tables:
                relevant_tables.add('airport')
    
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
