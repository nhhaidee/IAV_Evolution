import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize



def load_and_preprocess_data(segment, data_path='data', file_pattern_shap ='', file_pattern_rscu='', datatype= 'post2020', verbose=False):

    if datatype == 'post2020':
        shap_files = []
        rscu_files = []
        for i in range(10):
            shap_files.append(f'{data_path}/{segment}/{file_pattern_shap}{segment}_SHAP_Fold_{i}.csv')
            rscu_files.append(f'{data_path}/{segment}/{file_pattern_rscu}{segment}_RSCU_Fold_{i}.csv') #All_True_Label_Post2020_
        shap_df_list = [pd.read_csv(file) for file in shap_files]
        combined_shap_df = pd.concat(shap_df_list, ignore_index=True)
        rscu_df_list = [pd.read_csv(file) for file in rscu_files]
        combined_rscu_df = pd.concat(rscu_df_list, ignore_index=True)
    elif datatype == 'pre2020':
        combined_shap_df = pd.read_csv(f'{data_path}/{segment}/{file_pattern_shap}{segment}_SHAP.csv')
        combined_rscu_df = pd.read_csv(f'{data_path}/{segment}/{file_pattern_rscu}{segment}_RSCU.csv')
    # Codon columns are numeric features NOT in meta

    # --- Pre-cleaning & Renaming ---
    for df in [combined_shap_df, combined_rscu_df]:
        cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
        if cols_to_drop:
            df.drop(cols_to_drop, axis=1, inplace=True)

        rename_map = {}
        if 'True_Label' in df.columns: rename_map['True_Label'] = 'host_group'
        if 'Seq_ID' in df.columns: rename_map['Seq_ID'] = 'seq_ID'
        if 'Clade' in df.columns: rename_map['Clade'] = 'clade'
        if 'Subtype' in df.columns: rename_map['Subtype'] = 'subtype'
        if 'Pred_Label' in df.columns: rename_map['Pred_Label'] = 'pred_label'
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
    codon_cols = [col for col in combined_shap_df.columns if
                  col not in ['host_group', 'seq_ID', 'clade', 'subtype', 'pred_label']]
    fill_cols = ['clade', 'subtype']
    for col in fill_cols:
        if col in combined_shap_df.columns:
            combined_shap_df[col] = combined_shap_df[col].fillna('Unknown')
        if col in combined_rscu_df.columns:
            combined_rscu_df[col] = combined_rscu_df[col].fillna('Unknown')

    shap_df = combined_shap_df.groupby(['seq_ID', 'host_group', 'clade', 'subtype'], dropna=False)[codon_cols].mean().reset_index()
    rscu_df = combined_rscu_df.groupby(['seq_ID', 'host_group', 'clade', 'subtype'], dropna=False)[codon_cols].mean().reset_index()

    # --- Map Host Groups ---
    host_mapping = {0: 'Human', 1: 'Avian', 2: 'Non-human Mammal'}
    if 'host_group' in shap_df.columns and pd.api.types.is_numeric_dtype(shap_df['host_group']):
        shap_df['host_group'] = shap_df['host_group'].map(host_mapping)
    if 'host_group' in rscu_df.columns and pd.api.types.is_numeric_dtype(rscu_df['host_group']):
        rscu_df['host_group'] = rscu_df['host_group'].map(host_mapping)

    # --- Common Seqs Intersection ---
    common_seqs = set(shap_df['seq_ID']).intersection(set(rscu_df['seq_ID']))

    if len(common_seqs) == 0:
        print("Warning: No common sequences found after filtering!")
        return pd.DataFrame(), pd.DataFrame(), []

    shap_df = shap_df[shap_df['seq_ID'].isin(common_seqs)].reset_index(drop=True)
    rscu_df = rscu_df[rscu_df['seq_ID'].isin(common_seqs)].reset_index(drop=True)

    shap_df = shap_df.sort_values('seq_ID').reset_index(drop=True)
    rscu_df = rscu_df.sort_values('seq_ID').reset_index(drop=True)

    assert all(shap_df['seq_ID'] == rscu_df['seq_ID']), "Sequence IDs do not match after preprocessing."

    if verbose:
        print(f"Combined SHAP shape: {combined_shap_df.shape}")
        print(f"Combined RSCU shape: {combined_rscu_df.shape}")
        print(f"Cleaned SHAP shape: {shap_df.shape}")
        print(f"Cleaned RSCU shape: {rscu_df.shape}")
        if 'subtype' in shap_df.columns and 'host_group' in shap_df.columns:
            print("\nSequence Count by Subtype & Host Group:")
            # Group by both columns and count 'seq_ID'
            counts = shap_df.groupby(['subtype', 'host_group'])['seq_ID'].count().reset_index(name='count')
            # Pivot for cleaner viewing (Rows=Subtype, Cols=Host)
            try:
                counts_pivot = counts.pivot(index='subtype', columns='host_group', values='count').fillna(0).astype(int)
                # Sort by total sequences per subtype descending
                counts_pivot['Total'] = counts_pivot.sum(axis=1)
                counts_pivot = counts_pivot.sort_values('Total', ascending=False).head(20)

                print(counts_pivot.to_string())
                print("-" * 40)
            except Exception as e:
                print(counts)
        elif 'host_group' in shap_df.columns:
            print(f"Host group distribution:")
            print(shap_df['host_group'].value_counts())

    return shap_df, rscu_df, codon_cols

def fit_maz_percentile(mammalian_distances, lower_quantile=0.75, upper_quantile=0.95):
    """
    Estimate the Mammalian Adaptation Zone (MAZ) from the empirical
    distribution of phenotypic distances observed among mammalian-adapted strains.

    Parameters
    ----------
    mammalian_distances : array-like
        One-dimensional array of phenotypic distances for confirmed
        mammalian-adapted strains.
    lower_quantile : float, default=0.75
        Lower bound of the MAZ.
    upper_quantile : float, default=0.95
        Upper bound of the MAZ.

    Returns
    -------
    dict
        Dictionary containing the MAZ bounds and summary statistics.
    """
    x = np.asarray(mammalian_distances, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise ValueError("No valid mammalian distances were provided.")

    return {
        "method": "percentile",
        "n": int(x.size),
        "maz_lower": round(float(np.quantile(x, lower_quantile)),4),
        "maz_upper": round(float(np.quantile(x, upper_quantile)),4),
        "median_distance": round(float(np.median(x)), 4),
        "mean_distance": round(float(np.mean(x)),4),
        "sd_distance": round((float(np.std(x, ddof=1)) if x.size > 1 else np.nan), 4),
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
    }

def get_medoid_optimized(X, max_samples=50000, batch_size=1000, seed=42):
    """
    Computes the medoid for Cosine Distance with O(N) memory efficiency.
    
    Args:
        X (np.ndarray): Input array of shape (N, features).
        max_samples (int): Downsample threshold.
        batch_size (int): Size of chunks to process to avoid MemoryError.
        seed (int): Random seed for downsampling.
        
    Returns:
        np.ndarray: The medoid vector (real data point).
    """
    N = X.shape[0]
    if N == 0: return None
    if N == 1: return X[0]

    # 1. Downsample if necessary
    if N > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, max_samples, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    X_norm = normalize(X_sub, norm='l2', axis=1, copy=False)

    n_sub = X_norm.shape[0]
    sum_similarities = np.zeros(n_sub)
    
    for start in range(0, n_sub, batch_size):
        end = min(start + batch_size, n_sub)
        batch_sims = np.dot(X_norm[start:end], X_norm.T)
        sum_similarities[start:end] = batch_sims.sum(axis=1)

    medoid_idx = np.argmax(sum_similarities) # a “representative member” which an actual biological sequence
    return X_sub[medoid_idx]
    
def batch_cosine_dist(X_query, ref_vector):
    """
    Computes cosine distance between many query vectors and a single reference vector.
    Optimization: Uses dot product of normalized vectors instead of cdist.
    
    Args:
        X_query (np.ndarray): Shape (N, features)
        ref_vector (np.ndarray): Shape (features,) or (1, features)
        batch_size (int): Size of chunks (less critical here but good for consistency)
        
    Returns:
        np.ndarray: Array of cosine distances of shape (N,)
    """
    # 1. Ensure ref_vector is shape (1, features)
    ref_vector = ref_vector.reshape(1, -1)
    
    ref_norm = normalize(ref_vector, norm='l2', axis=1)
    
    X_query_norm = normalize(X_query, norm='l2', axis=1)
    
    similarities = X_query_norm.dot(ref_norm.T).flatten()

    similarities = np.clip(similarities, -1.0, 1.0)
    distances = 1.0 - similarities
    
    return distances