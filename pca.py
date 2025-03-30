# pca.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def apply_pca(expr_df, top_n=150, variance_threshold=0.9, is_scaler=False):
    # Select top N most variable genes
    var_df = expr_df.var().sort_values(ascending=False).head(top_n)
    top_genes = var_df.index.tolist()
    expr_top = expr_df[top_genes]

    # Scale and apply PCA
    if is_scaler:
        scaler = StandardScaler()
        expr_scaled = scaler.fit_transform(expr_top)
    else:
        scaler = None
        expr_scaled = expr_top
    
    pca = PCA(n_components=variance_threshold)
    expr_pca = pca.fit_transform(expr_scaled)

    pca_df = pd.DataFrame(expr_pca, index=expr_df.index,
                          columns=[f'PC{i+1}' for i in range(expr_pca.shape[1])])

    return pca_df, pca, scaler, top_genes
