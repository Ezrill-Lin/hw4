"""
Utility script to pre-compute ground truth records for dev set.
This speeds up evaluation during training.
"""

import os
import pickle
from utils import compute_records, read_queries

def create_ground_truth_records():
    """Pre-compute ground truth database records for dev set."""
    
    print("Creating ground truth records for dev set...")
    
    # Dev set
    dev_sql_path = 'data/dev.sql'
    dev_record_path = 'records/dev_gt_records.pkl'
    
    if not os.path.exists(dev_record_path):
        print(f"Computing records for {dev_sql_path}...")
        queries = read_queries(dev_sql_path)
        records, error_msgs = compute_records(queries)
        
        os.makedirs('records', exist_ok=True)
        with open(dev_record_path, 'wb') as f:
            pickle.dump((records, error_msgs), f)
        print(f"✓ Saved to {dev_record_path}")
    else:
        print(f"✓ {dev_record_path} already exists")
    
    # Also create ground_truth_dev.pkl if needed (alternate naming)
    alt_dev_record_path = 'records/ground_truth_dev.pkl'
    if not os.path.exists(alt_dev_record_path):
        print(f"Creating alternate naming: {alt_dev_record_path}...")
        if os.path.exists(dev_record_path):
            # Copy the file
            with open(dev_record_path, 'rb') as f:
                data = pickle.load(f)
            with open(alt_dev_record_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Saved to {alt_dev_record_path}")
    
    print("\n✅ Ground truth records ready!")

if __name__ == "__main__":
    create_ground_truth_records()
