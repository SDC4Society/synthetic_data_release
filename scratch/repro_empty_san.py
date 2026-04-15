import pandas as pd
from sanitisation_techniques.sanitiser_nhs import SanitiserNHS
from utils.constants import CATEGORICAL

# Mock metadata for MovieLens
metadata = {
    'columns': [
        {'name': '(no genres listed)', 'type': CATEGORICAL, 'i2s': ['0', '1']},
        {'name': 'Animation', 'type': CATEGORICAL, 'i2s': ['0', '1']},
        {'name': 'Documentary', 'type': CATEGORICAL, 'i2s': ['0', '1']},
        {'name': 'Film-Noir', 'type': CATEGORICAL, 'i2s': ['0', '1']},
        {'name': 'Horror', 'type': CATEGORICAL, 'i2s': ['0', '1']},
        {'name': 'IMAX', 'type': CATEGORICAL, 'i2s': ['0', '1']},
        {'name': 'Musical', 'type': CATEGORICAL, 'i2s': ['0', '1']},
        {'name': 'Mystery', 'type': CATEGORICAL, 'i2s': ['0', '1']},
        {'name': 'War', 'type': CATEGORICAL, 'i2s': ['0', '1']},
        {'name': 'Western', 'type': CATEGORICAL, 'i2s': ['0', '1']}
    ]
}

# Load actual data to see sparsity
data_path = 'data/movielens/ncols_10/eff_rank_best/eff_rank_best.csv'
try:
    df = pd.read_csv(data_path, index_col=0)
    print(f"Loaded {len(df)} records from {data_path}")
    
    # Take a sample of 1000 as in runconfig
    sample = df.sample(n=1000, random_state=42)
    
    quids = [c['name'] for c in metadata['columns']]
    
    # Try SanitiserNHS with t=2
    sanitizer = SanitiserNHS(metadata, nbins=10, thresh_rare=1, max_quantile=0.99, anonymity_set_size=2, quids=quids)
    print("Starting sanitization...")
    san_data = sanitizer.sanitise(sample)
    print(f"Sanitization completed. Sanitized data size: {len(san_data)}")
    
    from predictive_models.predictive_model import RandForestClassTask
    
    # Try training a utility task on the (potentially empty) sanitized data
    ut = RandForestClassTask(metadata, 'Mystery')
    print(f"Attempting to train {ut.__name__}...")
    ut.train(san_data)
    print("Training call completed successfully.")
    
    if len(san_data) == 0:
        print("Confirmed: Sanitized data is empty, but no crash occurred!")
        # Check evaluation on empty model
        acc = ut.evaluate(sample)
        print(f"Evaluation results (subset): {acc[:5]}")
        if all(a == 0 for a in acc):
            print("Verified: Evaluation returned 0 for untrained model.")

except Exception as e:
    import traceback
    print(f"Error during reproduction:\n{traceback.format_exc()}")
