import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from utils.constants import CATEGORICAL, ORDINAL, FLOAT, INTEGER

def create_preprocessing_pipeline(metadata, exclude_cols=None):
    """
    Creates an sklearn Pipeline that processes data according to the dataset metadata.
    
    Args:
        metadata: dict: The dataset metadata dict holding `columns`
        exclude_cols: list or str: Optional column name(s) to ignore during feature encoding.
        
    Returns:
        pipeline: sklearn.pipeline.Pipeline that transforms DataFrame to Numpy array.
    """
    if exclude_cols is None:
        exclude_cols = []
    elif isinstance(exclude_cols, str):
        exclude_cols = [exclude_cols]
        
    num_cols = []
    cat_cols = []
    cat_categories = []
    
    for cdict in metadata['columns']:
        name = cdict['name']
        if name in exclude_cols:
            continue
            
        dtype = cdict['type']
        if dtype in [FLOAT, INTEGER]:
            num_cols.append(name)
        elif dtype in [CATEGORICAL, ORDINAL]:
            cat_cols.append(name)
            cat_categories.append(cdict['i2s'])
            
    transformers = []
    
    if num_cols:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ])
        transformers.append(('num', num_pipeline, num_cols))
        
    if cat_cols:
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(categories=cat_categories, sparse_output=False, handle_unknown='ignore'))
        ])
        transformers.append(('cat', cat_pipeline, cat_cols))
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return Pipeline([
        ('preprocessor', preprocessor)
    ])
