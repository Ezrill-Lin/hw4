"""
Error Analysis Script for T5 Text-to-SQL Models
Analyzes and categorizes errors from fine-tuned and from-scratch models
"""

import pickle
import re
from collections import defaultdict
import sqlite3

def load_data(nl_file, sql_gt_file, sql_ft_file, sql_scr_file, record_gt_file, record_ft_file, record_scr_file):
    """Load all necessary data for error analysis"""
    # Load natural language queries
    with open(nl_file, 'r') as f:
        nl_queries = [line.strip() for line in f]
    
    # Load ground truth SQL
    with open(sql_gt_file, 'r') as f:
        sql_gt = [line.strip() for line in f]
    
    # Load fine-tuned predictions
    with open(sql_ft_file, 'r') as f:
        sql_ft = [line.strip() for line in f]
    
    # Load from-scratch predictions
    with open(sql_scr_file, 'r') as f:
        sql_scr = [line.strip() for line in f]
    
    # Load records (stored as tuple with [0] being list of record lists)
    with open(record_gt_file, 'rb') as f:
        records_gt_raw = pickle.load(f)
    
    with open(record_ft_file, 'rb') as f:
        records_ft_raw = pickle.load(f)
    
    with open(record_scr_file, 'rb') as f:
        records_scr_raw = pickle.load(f)
    
    # Extract the record lists from the tuple structure
    # records_*_raw is a tuple where first element is list of record lists
    records_gt = [set(map(str, r)) if r else set() for r in records_gt_raw[0]]
    records_ft = [set(map(str, r)) if r else set() for r in records_ft_raw[0]]
    records_scr = [set(map(str, r)) if r else set() for r in records_scr_raw[0]]
    
    return nl_queries, sql_gt, sql_ft, sql_scr, records_gt, records_ft, records_scr

def extract_tables(sql):
    """Extract table names from SQL query"""
    # Match FROM and JOIN clauses
    matches = re.findall(r'FROM\s+(\w+)|JOIN\s+(\w+)', sql.upper())
    tables = set(m[0] if m[0] else m[1] for m in matches)
    return tables

def extract_columns(sql):
    """Extract column selections from SQL query"""
    # Match SELECT clause
    match = re.search(r'SELECT\s+(.*?)\s+FROM', sql.upper())
    if match:
        return match.group(1).strip()
    return ""

def has_aggregation(sql):
    """Check if SQL has aggregation functions"""
    agg_funcs = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
    return any(func in sql.upper() for func in agg_funcs)

def has_group_by(sql):
    """Check if SQL has GROUP BY"""
    return 'GROUP BY' in sql.upper()

def has_order_by(sql):
    """Check if SQL has ORDER BY"""
    return 'ORDER BY' in sql.upper()

def extract_where_conditions(sql):
    """Extract WHERE conditions"""
    match = re.search(r'WHERE\s+(.*?)(?:GROUP|ORDER|$)', sql.upper())
    if match:
        return match.group(1).strip()
    return ""

def categorize_error(nl, sql_gt, sql_pred, records_gt, records_pred):
    """Categorize the type of error"""
    errors = []
    
    # Check if prediction is empty or error
    if not sql_pred or sql_pred.strip() == '':
        return ['empty_prediction']
    
    # Records match exactly - no error
    if records_gt == records_pred:
        return ['no_error']
    
    # Extract components
    tables_gt = extract_tables(sql_gt)
    tables_pred = extract_tables(sql_pred)
    
    cols_gt = extract_columns(sql_gt)
    cols_pred = extract_columns(sql_pred)
    
    where_gt = extract_where_conditions(sql_gt)
    where_pred = extract_where_conditions(sql_pred)
    
    # 1. Table/Schema Selection Errors
    if tables_gt != tables_pred:
        if len(tables_pred - tables_gt) > 0:
            errors.append('extra_tables')
        if len(tables_gt - tables_pred) > 0:
            errors.append('missing_tables')
        if not errors:  # Tables are different but same count
            errors.append('wrong_tables')
    
    # 2. Join/Relationship Errors
    if len(tables_gt) > 1 or len(tables_pred) > 1:
        # Check if join structure is different
        if 'JOIN' in sql_gt.upper() and 'JOIN' not in sql_pred.upper():
            errors.append('missing_join')
        elif 'JOIN' not in sql_gt.upper() and 'JOIN' in sql_pred.upper():
            errors.append('extra_join')
        elif tables_gt == tables_pred and records_gt != records_pred:
            # Same tables but different results - likely join condition issue
            errors.append('incorrect_join_condition')
    
    # 3. Column Selection Errors
    if 'DISTINCT' in cols_gt and 'DISTINCT' not in cols_pred:
        errors.append('missing_distinct')
    if cols_gt != cols_pred and not any(e in errors for e in ['extra_tables', 'missing_tables']):
        errors.append('wrong_columns')
    
    # 4. Aggregation Errors
    has_agg_gt = has_aggregation(sql_gt)
    has_agg_pred = has_aggregation(sql_pred)
    if has_agg_gt and not has_agg_pred:
        errors.append('missing_aggregation')
    elif not has_agg_gt and has_agg_pred:
        errors.append('extra_aggregation')
    elif has_agg_gt and has_agg_pred:
        # Check GROUP BY
        if has_group_by(sql_gt) and not has_group_by(sql_pred):
            errors.append('missing_group_by')
        # Different aggregation function
        agg_match_gt = re.findall(r'(COUNT|SUM|AVG|MIN|MAX)', sql_gt.upper())
        agg_match_pred = re.findall(r'(COUNT|SUM|AVG|MIN|MAX)', sql_pred.upper())
        if agg_match_gt != agg_match_pred:
            errors.append('wrong_aggregation_function')
    
    # 5. Filter/Predicate Errors
    if where_gt != where_pred:
        if where_gt and not where_pred:
            errors.append('missing_where')
        elif not where_gt and where_pred:
            errors.append('extra_where')
        else:
            errors.append('incorrect_where_condition')
    
    # 6. Ordering Errors
    if has_order_by(sql_gt) and not has_order_by(sql_pred):
        errors.append('missing_order_by')
    elif not has_order_by(sql_gt) and has_order_by(sql_pred):
        errors.append('extra_order_by')
    
    # 7. Subquery Errors
    if '(SELECT' in sql_gt.upper() and '(SELECT' not in sql_pred.upper():
        errors.append('missing_subquery')
    elif '(SELECT' not in sql_gt.upper() and '(SELECT' in sql_pred.upper():
        errors.append('extra_subquery')
    
    # If no specific error identified but records differ
    if not errors:
        errors.append('other_semantic_error')
    
    return errors

def analyze_errors(nl_queries, sql_gt, sql_ft, sql_scr, records_gt, records_ft, records_scr):
    """Perform comprehensive error analysis"""
    
    error_stats_ft = defaultdict(int)
    error_stats_scr = defaultdict(int)
    
    error_examples_ft = defaultdict(list)
    error_examples_scr = defaultdict(list)
    
    total_queries = len(nl_queries)
    
    for i in range(total_queries):
        # Analyze fine-tuned model
        errors_ft = categorize_error(nl_queries[i], sql_gt[i], sql_ft[i], 
                                     records_gt[i], records_ft[i])
        for error in errors_ft:
            error_stats_ft[error] += 1
            if len(error_examples_ft[error]) < 5:  # Keep top 5 examples
                error_examples_ft[error].append({
                    'nl': nl_queries[i],
                    'sql_gt': sql_gt[i],
                    'sql_pred': sql_ft[i],
                    'index': i
                })
        
        # Analyze from-scratch model
        errors_scr = categorize_error(nl_queries[i], sql_gt[i], sql_scr[i],
                                      records_gt[i], records_scr[i])
        for error in errors_scr:
            error_stats_scr[error] += 1
            if len(error_examples_scr[error]) < 5:  # Keep top 5 examples
                error_examples_scr[error].append({
                    'nl': nl_queries[i],
                    'sql_gt': sql_gt[i],
                    'sql_pred': sql_scr[i],
                    'index': i
                })
    
    return error_stats_ft, error_stats_scr, error_examples_ft, error_examples_scr, total_queries

def print_analysis(error_stats_ft, error_stats_scr, error_examples_ft, error_examples_scr, total_queries):
    """Print comprehensive error analysis"""
    
    print("="*80)
    print("ERROR ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total queries analyzed: {total_queries}\n")
    
    # Get all error types
    all_errors = set(error_stats_ft.keys()) | set(error_stats_scr.keys())
    all_errors.discard('no_error')  # Remove no_error for cleaner output
    
    # Sort by frequency
    sorted_errors = sorted(all_errors, 
                          key=lambda x: error_stats_ft.get(x, 0) + error_stats_scr.get(x, 0),
                          reverse=True)
    
    print(f"{'Error Type':<30} {'Fine-tuned':<20} {'From Scratch':<20}")
    print("-"*70)
    
    for error in sorted_errors:
        ft_count = error_stats_ft.get(error, 0)
        scr_count = error_stats_scr.get(error, 0)
        ft_pct = (ft_count / total_queries) * 100
        scr_pct = (scr_count / total_queries) * 100
        print(f"{error:<30} {ft_count:>4} ({ft_pct:>5.2f}%)     {scr_count:>4} ({scr_pct:>5.2f}%)")
    
    # Correct predictions
    ft_correct = error_stats_ft.get('no_error', 0)
    scr_correct = error_stats_scr.get('no_error', 0)
    print("-"*70)
    print(f"{'Correct (no error)':<30} {ft_correct:>4} ({(ft_correct/total_queries)*100:>5.2f}%)     {scr_correct:>4} ({(scr_correct/total_queries)*100:>5.2f}%)")
    
    print("\n" + "="*80)
    print("TOP ERROR EXAMPLES")
    print("="*80)
    
    # Print examples for top error types
    for error in sorted_errors[:10]:  # Top 10 error types
        print(f"\n{'='*80}")
        print(f"ERROR TYPE: {error}")
        print(f"{'='*80}")
        
        # Fine-tuned examples
        if error in error_examples_ft and len(error_examples_ft[error]) > 0:
            print(f"\nFINE-TUNED MODEL (occurs {error_stats_ft[error]} times, {(error_stats_ft[error]/total_queries)*100:.2f}%):")
            print("-"*80)
            for j, ex in enumerate(error_examples_ft[error][:2], 1):  # Show 2 examples
                print(f"\nExample {j} (Index {ex['index']}):")
                print(f"NL Query: {ex['nl']}")
                print(f"Ground Truth SQL: {ex['sql_gt']}")
                print(f"Predicted SQL: {ex['sql_pred']}")
        
        # From-scratch examples
        if error in error_examples_scr and len(error_examples_scr[error]) > 0:
            print(f"\nFROM-SCRATCH MODEL (occurs {error_stats_scr[error]} times, {(error_stats_scr[error]/total_queries)*100:.2f}%):")
            print("-"*80)
            for j, ex in enumerate(error_examples_scr[error][:2], 1):  # Show 2 examples
                print(f"\nExample {j} (Index {ex['index']}):")
                print(f"NL Query: {ex['nl']}")
                print(f"Ground Truth SQL: {ex['sql_gt']}")
                print(f"Predicted SQL: {ex['sql_pred']}")

def main():
    # File paths
    nl_file = 'data/dev.nl'
    sql_gt_file = 'data/dev.sql'
    sql_ft_file = 'results/t5_ft_optimized_schema_dev.sql'
    sql_scr_file = 'results/t5_scr_optimized_schema_scratch_dev.sql'
    record_gt_file = 'records/dev_gt_records.pkl'
    record_ft_file = 'records/t5_ft_optimized_schema_dev.pkl'
    record_scr_file = 'records/t5_scr_optimized_schema_scratch_dev.pkl'
    
    print("Loading data...")
    nl_queries, sql_gt, sql_ft, sql_scr, records_gt, records_ft, records_scr = load_data(
        nl_file, sql_gt_file, sql_ft_file, sql_scr_file,
        record_gt_file, record_ft_file, record_scr_file
    )
    
    print(f"Loaded {len(nl_queries)} queries")
    print("Analyzing errors...")
    
    error_stats_ft, error_stats_scr, error_examples_ft, error_examples_scr, total_queries = analyze_errors(
        nl_queries, sql_gt, sql_ft, sql_scr, records_gt, records_ft, records_scr
    )
    
    print_analysis(error_stats_ft, error_stats_scr, error_examples_ft, error_examples_scr, total_queries)
    
    # Save detailed results
    print("\n" + "="*80)
    print("Saving detailed analysis to error_analysis_results.txt...")
    with open('error_analysis_results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED ERROR ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        # Write statistics
        all_errors = set(error_stats_ft.keys()) | set(error_stats_scr.keys())
        all_errors.discard('no_error')
        sorted_errors = sorted(all_errors,
                              key=lambda x: error_stats_ft.get(x, 0) + error_stats_scr.get(x, 0),
                              reverse=True)
        
        f.write(f"{'Error Type':<30} {'Fine-tuned':<20} {'From Scratch':<20}\n")
        f.write("-"*70 + "\n")
        for error in sorted_errors:
            ft_count = error_stats_ft.get(error, 0)
            scr_count = error_stats_scr.get(error, 0)
            ft_pct = (ft_count / total_queries) * 100
            scr_pct = (scr_count / total_queries) * 100
            f.write(f"{error:<30} {ft_count:>4} ({ft_pct:>5.2f}%)     {scr_count:>4} ({scr_pct:>5.2f}%)\n")
        
        # Write all examples
        for error in sorted_errors:
            f.write(f"\n{'='*80}\n")
            f.write(f"ERROR TYPE: {error}\n")
            f.write(f"{'='*80}\n")
            
            if error in error_examples_ft:
                f.write(f"\nFINE-TUNED MODEL EXAMPLES:\n")
                for j, ex in enumerate(error_examples_ft[error], 1):
                    f.write(f"\nExample {j}:\n")
                    f.write(f"NL: {ex['nl']}\n")
                    f.write(f"GT: {ex['sql_gt']}\n")
                    f.write(f"Pred: {ex['sql_pred']}\n")
            
            if error in error_examples_scr:
                f.write(f"\nFROM-SCRATCH MODEL EXAMPLES:\n")
                for j, ex in enumerate(error_examples_scr[error], 1):
                    f.write(f"\nExample {j}:\n")
                    f.write(f"NL: {ex['nl']}\n")
                    f.write(f"GT: {ex['sql_gt']}\n")
                    f.write(f"Pred: {ex['sql_pred']}\n")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
