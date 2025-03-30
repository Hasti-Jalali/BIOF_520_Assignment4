# imputation.py

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def knn_impute(external_clinical_df, external_expr, clinical_df, exprs_df,
               comb_feature, feature_col, top_genes, cat_cols, n_neighbors=3):
    exter_feature_clin = [i for i in external_clinical_df.columns if i in feature_col]
    exter_feature_expr = [i for i in external_expr.columns if i in top_genes]

    X_train_sim = clinical_df[exter_feature_clin].join(exprs_df[exter_feature_expr])
    X_ext_sim = external_clinical_df[exter_feature_clin].join(external_expr[exter_feature_expr])

    X_train_sim.columns = X_train_sim.columns.astype(str)
    X_ext_sim.columns = X_ext_sim.columns.astype(str)
    
    nn_model = NearestNeighbors(n_neighbors=n_neighbors)
    nn_model.fit(X_train_sim.fillna(0))

    missing_cols = [i for i in (feature_col + top_genes) if i not in comb_feature]

    for i in external_clinical_df.index:
        row = X_ext_sim.loc[i]
        row_filled = row.fillna(0).values.reshape(1, -1)
        _, indices = nn_model.kneighbors(row_filled)
        nearest_indices = X_train_sim.iloc[indices[0]].index

        for col in missing_cols:
            if col in feature_col:
                target_df = clinical_df
                target_out_df = external_clinical_df
            elif col in top_genes:
                target_df = exprs_df
                target_out_df = external_expr
            else:
                continue

            if (col not in target_out_df.columns) or pd.isna(target_out_df.at[i, col]):
                neighbor_vals = target_df.loc[nearest_indices, col]
                if target_df[col].dtype == 'object' or col in cat_cols:
                    fill_val = neighbor_vals.mode().iloc[0]
                else:
                    fill_val = neighbor_vals.mean()
                target_out_df.at[i, col] = fill_val
    return external_clinical_df, external_expr
