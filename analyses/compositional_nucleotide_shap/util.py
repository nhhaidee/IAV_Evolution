import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Map integers to nucleotide characters (channel order must match data)
nt_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}


def get_shap_composition_vectorized(X_onehot, X_shap, species_index, is_sum=False):
    """
    X_onehot: (n_seq, C, L)
        One-hot encoded nucleotide sequences.
        C = 5 channels (A, C, G, T, N)
        L = sequence length

    X_shap: (n_seq, C, L, n_species)
        SHAP values for each nucleotide channel and each species.

    species_index: int
        Which species/class to extract SHAP values for (0, 1, or 2).

    Returns:
        raw_sum: dict of nucleotide ? (n_seq,) SHAP totals
        pad_sum: (n_seq,) SHAP wasted on padding
    """

    # ---------------------------------------------------------
    # 1. Select SHAP values for the chosen species
    # X_shap:        (n_seq, C, L, n_species)
    # shap_for_species: (n_seq, C, L)
    # ---------------------------------------------------------
    shap_for_species = X_shap[..., species_index]

    # ---------------------------------------------------------
    # 2. Optionally sum SHAP across channels
    # If is_sum=False:
    #     shap_values_for_species: (n_seq, C, L)
    # If is_sum=True:
    #     shap_values_for_species: (n_seq, 1, L)
    # ---------------------------------------------------------
    if is_sum:
        shap_values_for_species = np.sum(shap_for_species, axis=1, keepdims=True)
    else:
        shap_values_for_species = shap_for_species

    # ---------------------------------------------------------
    # 3. Identify padding positions
    # X_onehot: (n_seq, C, L)
    # np.sum(..., axis=1) ? (n_seq, L)
    # non_padding_mask: (n_seq, L) boolean
    # padding_mask:     (n_seq, L) boolean
    # ---------------------------------------------------------
    non_padding_mask = np.sum(X_onehot, axis=1) > 0
    padding_mask = ~non_padding_mask

    # ---------------------------------------------------------
    # 4. Compute SHAP at padding positions
    # summed_shaps_across_channels: (n_seq, L)
    # pad_scores_per_seq:           (n_seq,)
    # ---------------------------------------------------------
    summed_shaps_across_channels = np.sum(shap_for_species, axis=1)
    pad_scores_per_seq = np.sum(summed_shaps_across_channels * padding_mask, axis=1)

    # ---------------------------------------------------------
    # 5. Keep only SHAP for the actual nucleotide
    # shap_values_for_species: (n_seq, C, L)
    # X_onehot:                (n_seq, C, L)
    # ? actual_shap_scores:    (n_seq, C, L)
    #
    # Multiply by non_padding_mask[:, None, :] to zero out padding:
    # non_padding_mask[:, None, :]: (n_seq, 1, L)
    # ---------------------------------------------------------
    actual_shap_scores = shap_values_for_species * X_onehot
    actual_shap_scores = actual_shap_scores * non_padding_mask[:, np.newaxis, :]

    # ---------------------------------------------------------
    # 6. Sum SHAP across sequence positions
    # actual_shap_scores: (n_seq, C, L)
    # total_scores_by_nt: (n_seq, C)
    # ---------------------------------------------------------
    total_scores_by_nt = np.sum(actual_shap_scores, axis=2)

    # ---------------------------------------------------------
    # 7. Build output dictionary
    # Each nt_char gets an array of shape (n_seq,)
    # pad_sum is also (n_seq,)
    # ---------------------------------------------------------
    result = {
        'raw_sum': {nt_char: total_scores_by_nt[:, i] for i, nt_char in nt_map.items()},
        'pad_sum': pad_scores_per_seq,
    }
    return result
    

def shap_distribution_plot(
    ax,
    shap_scores,
    ylim=None,
    use_boxen=True
):
    """
    Publication-quality SHAP score distribution plot per nucleotide.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    shap_scores : dict
        { 'A': [...], 'C': [...], ..., 'Pad': [...] }
    ylim : tuple or None
        (ymin, ymax) for consistency across panels
    use_boxen : bool
        Use boxenplot (large-N) or boxplot (small-N)
    """

    data = []
    present_nts = []

    for k, v in shap_scores.items():
        if len(v) > 0:
            data.append(
                pd.DataFrame({
                    'SHAP Score': v,
                    'Nucleotide': k
                })
            )
            present_nts.append(k)

    if not data:
        ax.set_axis_off()
        return

    df_long = pd.concat(data, ignore_index=True)

    # Fixed canonical order
    base_order = ['A', 'C', 'G', 'T', 'N', 'Pad']
    plot_order = [nt for nt in base_order if nt in present_nts]

    # Muted, colorblind-safe palette
    palette = {
        'A': 'green',
        'C': 'blue',
        'G': 'orange',
        'T': 'red',
        'N': 'black',
        'Pad': 'gray'
    }
    palette = {k: palette[k] for k in plot_order}

    plot_kwargs = dict(
        ax=ax,
        data=df_long,
        x='Nucleotide',
        y='SHAP Score',
        order=plot_order,
        palette=palette,
        linewidth=1.0,
        showfliers=False
    )

    sns.boxenplot(
        **plot_kwargs,
        hue='Nucleotide',
        legend=False
    )

    # Zero reference line
    ax.axhline(
        0,
        color='0.3',
        linestyle='--',
        linewidth=1.4,
        zorder=0
    )

    # Axis styling
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')

    if ylim is not None:
        ax.set_ylim(ylim)
        
    
    ax.yaxis.set_major_locator(
            MaxNLocator(nbins=6)   # try 3 if you want even cleaner
        )


    ax.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False
    )
    ax.tick_params(axis='y', labelsize=20)

    # Clean spines
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
