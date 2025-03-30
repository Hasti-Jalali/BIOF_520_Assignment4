# external_validation.py

import pandas as pd
from helpers import print_model_report

def run_external_validation(models, external_df_final, y_true_ext):
    for name, model in models.items():
        y_pred = model.predict(external_df_final)
        y_proba = model.predict_proba(external_df_final)[:, 1]
        print_model_report(f"{name} (External)", y_true_ext, y_pred, y_proba)
