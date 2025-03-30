# helpers.py

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

def map_clinical_categories(df, cat_map):
    for col, mapping in cat_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df

def print_model_report(name, y_true, y_pred, y_proba):
    print(f"\n📊 {name} Evaluation")
    print(classification_report(y_true, y_pred))
    print("AUC:", roc_auc_score(y_true, y_proba))

def clean_labels(df, label_col='Recurrence'):
    df[label_col] = df[label_col].replace(-2147483648, np.nan)
    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)
    return df
