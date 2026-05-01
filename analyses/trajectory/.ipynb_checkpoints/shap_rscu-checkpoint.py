import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.lines import Line2D
from scipy.stats import false_discovery_control
from scipy.stats import pearsonr, ttest_ind, spearmanr
import matplotlib.patches as patches
import numpy as np
from scipy.stats import skew, wilcoxon, permutation_test
import math
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef, classification_report

def get_amino_acid(codon):
    """Convert DNA codon to amino acid (single letter code)"""

    genetic_code = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }

    return genetic_code.get(codon, 'X')

def get_codon_type(codon):
    gc_count = codon.count('G') + codon.count('C')
    return 'G/C-rich' if gc_count >= 2 else 'A/T-rich'

def codon_category_map(codons):
    """
    Map each codon to 'A/T-rich' or 'G/C-rich' by nucleotide composition.
    """
    cats = {}
    for c in codons:
      c2 = c.upper()
      at = c2.count('A') + c2.count('T')
      gc = c2.count('G') + c2.count('C')
      cats[c] = 'A/T-rich' if at > gc else 'G/C-rich'
    return cats

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

def analyze_codon_shap_trends(shap_df, codon_cols, codon_categories):

    # --- 1. Calculate Mean SHAP Scores for Each Host Group ---

    shap_df_avian = shap_df[shap_df['host_group'] == 'Avian'].reset_index(drop=True)
    shap_df_human = shap_df[shap_df['host_group'] == 'Human'].reset_index(drop=True)
    shap_df_mammal = shap_df[shap_df['host_group'] == 'Non-human Mammal'].reset_index(drop=True)

    mean_shap_avian = shap_df_avian[codon_cols].mean().to_dict()
    mean_shap_human = shap_df_human[codon_cols].mean().to_dict()
    mean_shap_mammal = shap_df_mammal[codon_cols].mean().to_dict()

    trend_data = []

    # --- 2. Determine the Trend for Each Codon ---
    for codon in codon_cols:
        s_avian = mean_shap_avian.get(codon, 0)
        s_human = mean_shap_human.get(codon, 0)
        s_mammal = mean_shap_mammal.get(codon, 0)
        cat = codon_categories.get(codon)
        trend = 'Other'  # Default trend

        # Check for monotonic trends
        if s_avian < s_human < s_mammal:
            trend = 'Increasing'
        elif s_avian > s_human > s_mammal:
            trend = 'Decreasing'
        # Check for peaked or valley trends
        elif s_avian < s_human and s_human > s_mammal:
            trend = 'Peaked in Human'
        elif s_avian > s_human and s_human < s_mammal:
            trend = 'Valley in Human'

        trend_data.append({
            'codon': codon,
            'shap_avian': s_avian,
            'shap_human': s_human,
            'shap_mammal': s_mammal,
            'trend': trend,
            'category': cat
        })

    trend_df = pd.DataFrame(trend_data)

    return trend_df

def codon_trajectory_plot_grid(
        trend_dfs_dict,
        codon_categories,
        segments=["01_PB2", "02_PB1", "03_PA", "04_HA",
                  "05_NP", "06_NA", "07_MP", "08_NS"],
        save_path=None
):
    """
    Creates a 2x4 grid of codon SHAP trajectories.
    Legend is placed in the top-left of the first subplot.
    """

    n_segments = len(segments)
    n_rows, n_cols = 2, 4

    if n_segments > n_rows * n_cols:
        raise ValueError("This function assumes at most 8 segments for a 2x4 grid.")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharey=True)
    axes = np.array(axes).reshape(n_rows, n_cols)

    hosts = ['Human', 'Avian', 'Non-human Mammal']
    for idx, segment in enumerate(segments):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        trend_df = trend_dfs_dict.get(segment, None)

        if trend_df is None or trend_df.empty:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax.set_title(segment, fontsize=14, fontweight='bold')
            continue

        # Plot Trajectories
        for _, row_df in trend_df.iterrows():
            cat = codon_categories.get(row_df['codon'], 'Other')
            color = 'blue' if cat == 'G/C-rich' else 'orange'
            ax.plot(
                hosts,
                [row_df['shap_human'], row_df['shap_avian'], row_df['shap_mammal']],
                color=color, alpha=0.6, linewidth=1.5
            )

        # Formatting
        if segment == '07_MP':
            ax.set_title('07_M', fontsize=18, pad=12)
        else:
            ax.set_title(segment, fontsize=18, pad=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=0, labelsize=9)

        # Only leftmost column gets y-label
        if col == 0:
            ax.set_ylabel('SHAP Values', fontsize=14)

    # Hide any unused axes if fewer than 8 segments
    for idx in range(n_segments, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    # --- Legend (Placed in First Subplot) ---
    legend_handles = [
        Line2D([0], [0], color='blue', lw=2, label='G/C-rich'),
        Line2D([0], [0], color='orange', lw=2, label='A/T-rich')
    ]
    fig.legend(
        handles=legend_handles,
        #title="Subtype",
        loc='upper center',
        bbox_to_anchor=(0.5, 1.03), # Anchored above the plots
        ncol=len(legend_handles),  # Horizontal layout
        frameon=True,
        fontsize=14
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compute_rscu_shap_correlations(
    shap_df,
    rscu_df,
    codon_cols,
    codon_categories,
    host_col='host_group',
    seq_col='seq_ID',
    alpha=0.05,
    method='spearman'
):
    """
    Compute per-codon correlation (SHAP vs RSCU).
    """
    if method not in {'pearson', 'spearman'}:
        raise ValueError("method must be 'pearson' or 'spearman'")

    corr_func = pearsonr if method == 'pearson' else spearmanr

    # Ensure common sequences and align
    assert shap_df[seq_col].equals(rscu_df[seq_col]), "Sequence IDs do not match between SHAP and RSCU DataFrames."

    rows = []
    host_groups = shap_df[host_col].dropna().unique()
    for host in host_groups:
        shap_h = shap_df[shap_df[host_col] == host]
        rscu_h = rscu_df[rscu_df[seq_col].isin(shap_h[seq_col])]
        # Merge
        merged = shap_h[[seq_col] + codon_cols].merge(
            rscu_h[[seq_col] + codon_cols],
            on=seq_col, suffixes=('_shap', '_rscu'))

        for c in codon_cols:
            x = merged[f'{c}_shap'].values
            y = merged[f'{c}_rscu'].values
            mask = np.isfinite(x) & np.isfinite(y)
            #print (host, c, x.shape, y.shape, mask.shape)
            # Validity Check
            if (mask.sum() < 3 or
                np.std(x[mask]) == 0 or  # Safer/Faster than np.all logic
                np.std(y[mask]) == 0):
                r, p = np.nan, np.nan
            else:
                try:
                    res = corr_func(x[mask], y[mask])
                    r, p = res[0], res[1]
                except Exception:
                    r, p = np.nan, np.nan
            # Construct Row
            row_data = {'host': host, 'codon': c, 'r': r, 'p': p}

            rows.append(row_data)

    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        return corr_df
    # Sort
    sort_cols = ['host', 'r']
    corr_df = corr_df.sort_values(sort_cols, ascending=[True, False]).reset_index(drop=True)

    # FDR Correction (Fixed)
    group_cols = ['host']
    if 'segment' in corr_df.columns:
        group_cols.append('segment')

    corr_df['q'] = np.nan
    corr_df['sig'] = False

    for keys, sub in corr_df.groupby(group_cols, dropna=False):
        # We need to map back to the original DataFrame indices
        idx_all = sub.index
        valid_mask = np.isfinite(sub['p'])
        if valid_mask.sum() > 0:
            p_vals = sub.loc[valid_mask, 'p'].values
            q_vals = false_discovery_control(p_vals, method='bh')
            sig_vals = q_vals <= alpha
            # FIX: Use idx_all[valid_mask] to target ONLY the rows that had valid p-values
            corr_df.loc[idx_all[valid_mask], 'q'] = q_vals
            corr_df.loc[idx_all[valid_mask], 'sig'] = sig_vals
    corr_df['codon type'] = corr_df['codon'].apply(lambda x: codon_categories[x])
    return corr_df


def volcano_plot_grid(
    segment_dfs,
    codon_categories,
    hosts=["Avian", "Non-human Mammal", "Human"],
    segments=["01_PB2", "02_PB1", "03_PA", "04_HA", "05_NP", "06_NA", "07_MP", "08_NS"],
    alpha=0.05,
    point_size=20,
    grid_shape=(8, 3),   # changed from implicit 3x8 to configurable 6x4
    save_path=None,
    count_significant_only=False,
    sig_threshold=None,
):
    """
    Creates a grid of volcano plots, defaulting to 6x4.
    Points with q=0 are moved to a ceiling and enclosed in a green bounding box.

    Parameters
    ----------
    segment_dfs : dict
        Dictionary of DataFrames keyed by segment name.
    codon_categories : dict
        Mapping of codon to 'A/T-rich' or 'G/C-rich'.
    hosts : list
        Host groups.
    segments : list
        Segment names.
    alpha : float
        Significance threshold.
    point_size : int
        Scatter point size.
    grid_shape : tuple
        Figure layout as (n_rows, n_cols). Default is (6, 4).
    save_path : str, optional
        Path to save the figure.
    """
    epsilon = 1.0e-100
    grid_rows, grid_cols = grid_shape

    # Build panel order: one panel per (host, segment)
    #panel_specs = [(host, segment) for host in hosts for segment in segments]
    panel_specs = [(segment, host) for segment in segments for host in hosts]
    n_panels = len(panel_specs)
    print (panel_specs)

    if grid_rows * grid_cols < n_panels:
        raise ValueError(
            f"grid_shape={grid_shape} only has {grid_rows * grid_cols} slots, "
            f"but {n_panels} panels are required."
        )

    # 1. Pre-calculate global max y from all non-zero q values
    global_max_y = 0
    for seg_df in segment_dfs.values():
        if seg_df is not None and not seg_df.empty:
            non_zeros = seg_df.loc[seg_df["q"] != 0, "q"]
            if not non_zeros.empty:
                safe_q = non_zeros.replace(0, epsilon).fillna(1.0)
                current_max = (-np.log10(safe_q)).max()
                global_max_y = max(global_max_y, current_max)

    if global_max_y == 0:
        global_max_y = 5

    ceiling_y = global_max_y * 1.15
    plot_ymax = ceiling_y * 1.25

    # Figure sizing tuned for 6x4 readability
    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(4.5 * grid_cols, 3.8 * grid_rows),
        sharex=False,
        sharey=False
    )
    axes = np.atleast_1d(axes).flatten()

    color_map = {"A/T-rich": "orange", "G/C-rich": "blue"}

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= n_panels:
            ax.set_visible(False)
            continue

        segment, host = panel_specs[ax_idx]
        print (ax_idx, segment, host)
        current_segment_df = segment_dfs.get(segment, None)

        # Handle missing segment dataframe
        if current_segment_df is None:
            ax.text(0.5, 0.5, "Data Missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{host} | {'07_M' if segment == '07_MP' else segment}", fontsize=11)
            ax.set_ylim(-50, plot_ymax)
            ax.set_xlabel(r"Spearman $\rho$", fontsize=10)
            ax.set_ylabel(r"-log$_{10}$(q)", fontsize=10)
            continue

        sub = current_segment_df[current_segment_df["host"] == host].copy()

        if sub.empty:
            ax.text(0.5, 0.5, "No Host Data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{host} | {'07_M' if segment == '07_MP' else segment}", fontsize=11)
            ax.set_ylim(-50, plot_ymax)
            ax.set_xlabel(r"Spearman $\rho$", fontsize=10)
            ax.set_ylabel(r"-log$_{10}$(q)", fontsize=10)
            continue

        # Data transformation
        sub["is_zero_q"] = sub["q"] == 0
        sub["q"] = sub["q"].replace(0, epsilon).fillna(1.0)
        sub["neglog10q"] = -np.log10(sub["q"])
        sub.loc[sub["is_zero_q"], "neglog10q"] = ceiling_y

        sub["category"] = sub["codon"].map(codon_categories)
        colors = sub["category"].map(color_map).fillna("gray")

        # Plot
        ax.scatter(
            sub["r"],
            sub["neglog10q"],
            c=colors,
            s=point_size,
            alpha=0.8,
            edgecolors="w",
            linewidths=0.5
        )

        ax.axhline(-np.log10(alpha), ls="--", c="k", lw=1, alpha=0.4)
        ax.axvline(0, ls="-", c="k", lw=1, alpha=0.6)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_ylim(-50, plot_ymax)

        # Add q-value = 0.05 label after limits are set
        q_05_y = -np.log10(alpha)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        x_right = x1 - 0.02 * (x1 - x0)
        y_above = q_05_y + 0.02 * (y1 - y0)

        ax.text(
            x_right,
            y_above,
            "q-value = 0.05",
            va="bottom",
            ha="right",
            fontsize=10,
            alpha=0.8,
            color="k",
            transform=ax.transData
        )

        # Bounding boxes for q = 0 clusters
        if sub["is_zero_q"].any():
            zeros = sub[sub["is_zero_q"]]
            box_h = plot_ymax * 0.08

            def add_box_annotation(cluster_df):
                r_vals = cluster_df["r"]
                r_min, r_max = r_vals.min(), r_vals.max()

                x0, x1 = ax.get_xlim()
                x_span = x1 - x0
                x_pad = x_span * 0.03

                rect_x = r_min - x_pad
                rect_y = ceiling_y - (box_h / 2)
                rect_w = (r_max - r_min) + (2 * x_pad)

                rect = patches.Rectangle(
                    (rect_x, rect_y),
                    rect_w,
                    box_h,
                    linewidth=1,
                    edgecolor="green",
                    facecolor="none",
                    linestyle="--",
                    zorder=10
                )
                ax.add_patch(rect)

                ax.text(
                    r_min + (r_max - r_min) / 2,
                    #rect_y + box_h + (plot_ymax * 0.01),
                    rect_y - (plot_ymax * 0.05),  # slightly below rect bottom
                    "q-value = 0",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="green",
                    fontweight="bold"
                )

            pos_zeros = zeros[zeros["r"] > 0]
            neg_zeros = zeros[zeros["r"] < 0]

            if not pos_zeros.empty:
                add_box_annotation(pos_zeros)
            if not neg_zeros.empty:
                add_box_annotation(neg_zeros)
        
        if count_significant_only:
            count_df = sub[sub["sig"]]
        else:
            count_df = sub

        left = count_df[count_df["r"] < 0]
        right = count_df[count_df["r"] > 0]

        left_at = (left["category"] == "A/T-rich").sum()
        left_gc = (left["category"] == "G/C-rich").sum()
        right_at = (right["category"] == "A/T-rich").sum()
        right_gc = (right["category"] == "G/C-rich").sum()

        label_suffix = " (sig)" if count_significant_only else ""

        ax.text(
            0.02, 0.98,
            f"A/T: {left_at}{label_suffix}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10, fontweight="bold",
            color="orange"
        )
        ax.text(
            0.02, 0.92,
            f"G/C: {left_gc}{label_suffix}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10, fontweight="bold",
            color="blue"
        )

        ax.text(
            0.98, 0.98,
            f"A/T: {right_at}{label_suffix}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10, fontweight="bold",
            color="orange"
        )
        ax.text(
            0.98, 0.92,
            f"G/C: {right_gc}{label_suffix}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10, fontweight="bold",
            color="blue"
        )
        # Panel title
        title_segment = "07_M" if segment == "07_MP" else segment
        ax.set_title(f"{title_segment} | {host}", fontsize=16)
        ax.set_xlabel(r"Spearman $\rho$", fontsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        if ax_idx%3==0:
            ax.set_ylabel(r"-log$_{10}$(q)", fontsize=12)

    # Legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="G/C-rich",
               markerfacecolor=color_map["G/C-rich"], markersize=12),
        Line2D([0], [0], marker="o", color="w", label="A/T-rich",
               markerfacecolor=color_map["A/T-rich"], markersize=12)
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=12
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()