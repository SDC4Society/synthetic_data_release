import pandas as pd
from preprocess_common.pipeline import create_preprocessing_pipeline
from utils.datagen import load_local_data_as_df

df, m = load_local_data_as_df('data/texas')
pipeline = create_preprocessing_pipeline(m, exclude_cols=['DISCHARGE'])
arr = pipeline.fit_transform(df)
print(arr.shape)
