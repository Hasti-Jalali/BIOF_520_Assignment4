# preprocessing.py

import pandas as pd
from helpers import map_clinical_categories

def preprocess_clinical(df, num_cols, cat_cols, median_values=None, mode_values=None):
    df = df.copy()

    # Fill numeric
    if median_values is None:
        median_values = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(median_values)

    # Fill categorical
    if mode_values is None:
        mode_values = {col: df[col].mode()[0] for col in cat_cols}
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(mode_values[col])

    return df, median_values, mode_values

def drop_constant_columns(df):
    return df.loc[:, df.nunique() > 1]
