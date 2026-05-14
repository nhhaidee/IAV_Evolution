import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold
from util import get_shap_composition_vectorized, shap_distribution_plot, nt_map
import matplotlib.patches as mpatches


def main():
    segments = ['01_PB2', '02_PB1', '03_PA', '04_HA', '05_NP', '06_NA', '07_MP', '08_NS']
    n_rows = 3 * 2
    n_cols = int(len(segments) /2)

    fig_shap, axes_shap = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows), sharey=False)
    
    legend_handles = [
      mpatches.Patch(color='green', label='A'),
      mpatches.Patch(color='blue', label='C'),
      mpatches.Patch(color='orange', label='G'),
      mpatches.Patch(color='red', label='T'),
      mpatches.Patch(color='black', label='N'),
      mpatches.Patch(color='gray', label='Padding'),
    ]


    os.makedirs('Figs', exist_ok=True)

    for col_idx, segment in enumerate(segments):

        base_path = '/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new'
        train_data_path = f'{base_path}/data/{segment}'
        shap_data_path = f'{base_path}/shap/{segment}'
        pred_data_path = f'/fs/vnas_Hcfia/orph/hon000/part2/reassortment/{segment}'

        try:
            X_train = np.load(f'{train_data_path}/X_train_onehot.npy')  # (n_seq, C, L)
            y_train = np.load(f'{train_data_path}/y_train.npy')         # (n_seq,)
            y_pred = np.load(f'{pred_data_path}/pre_2020_pred_cal.npy') # (n_seq,)
        except FileNotFoundError as e:
            print(f"Skipping {segment}: Data file not found ({e})")
            continue

        n_seq = X_train.shape[0]
        res_len = X_train.shape[1]  # expected C=5
        seq_len = X_train.shape[2]  # L

        # Prepare container for OOF SHAP: (n_seq, C, L, 3)
        try:
            shap_oof = np.zeros((n_seq, res_len, seq_len, 3), dtype=np.float32)
            splitter = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=0)
            n_chunks = 5

            splits = list(splitter.split(np.zeros(len(y_train)), y_train))
            if len(splits) != 10:
                print(f"[WARN] Expected 10 folds, got {len(splits)}")

            for fold, (_, val_indices) in enumerate(splits):
                shap_files = [f'{shap_data_path}/{segment}_training_shap_values_Fold_{fold}_chunk_{chunk}.npy'
                              for chunk in range(n_chunks)]

                chunk_arrays = []
                for f in shap_files:
                    if not os.path.exists(f):
                        raise FileNotFoundError(f"Missing SHAP file: {f}")
                    arr = np.load(f)
                    chunk_arrays.append(arr)
                shap_values = np.concatenate(chunk_arrays, axis=0)

                expected_shape = (len(val_indices), res_len, seq_len, 3)
                if shap_values.shape != expected_shape:
                    print(f"[WARN] {segment} Fold {fold}: got {shap_values.shape}, expected {expected_shape}")

                shap_oof[val_indices, ...] = shap_values

        except Exception as e:
            print(f"Error loading SHAP for {segment}: {e}")
            continue

        # Aggregation holders
        agg_data = {
            'Human': {'raw_sum': {k: [] for k in list(nt_map.values()) + ['Pad']}},
            'Avian': {'raw_sum': {k: [] for k in list(nt_map.values()) + ['Pad']}},
            'Non-human Mammal': {'raw_sum': {k: [] for k in list(nt_map.values()) + ['Pad']}}
        }
        
        species_idx = {'Human': 0, 'Avian': 1, 'Non-human Mammal': 2}

        # Only include correctly predicted samples for each species
        indices_mask = {
            'Human': (y_pred == 0) & (y_pred == y_train),
            'Avian': (y_pred == 1) & (y_pred == y_train),
            'Non-human Mammal': (y_pred == 2) & (y_pred == y_train)
        }

        for species, mask in indices_mask.items():
            if np.sum(mask) == 0:
                continue

            sp_onehot = X_train[mask]
            sp_shap = shap_oof[mask]

            comp = get_shap_composition_vectorized(sp_onehot, sp_shap, species_idx[species])

            # raw sums & pad
            for nt in nt_map.values():
                agg_data[species]['raw_sum'][nt].extend(comp['raw_sum'][nt])
            agg_data[species]['raw_sum']['Pad'].extend(comp['pad_sum'])


        # Plot per species row
        for species_name, species_id in species_idx.items():
            segment_name = segment if segment != '07_MP' else '07_M'
            # Original panels
            if col_idx <=3:
              shap_distribution_plot(ax=axes_shap[species_id, col_idx],
                                     shap_scores=agg_data[species_name]['raw_sum'])
                                     
              axes_shap[species_id, col_idx].set_title(f'{species_name} | {segment_name}', fontsize=18, pad=10)
            else:
              shap_distribution_plot(ax=axes_shap[species_id + 3, col_idx%4],
                                     shap_scores=agg_data[species_name]['raw_sum'])
                                     
              axes_shap[species_id + 3, col_idx%4].set_title(f'{species_name} | {segment_name}', fontsize=18, pad=10)

    fig_shap.legend(
      handles=legend_handles,
      loc='upper center',
      ncol=6,
      frameon=False,
      fontsize=20,
      bbox_to_anchor=(0.5, 1.02)
    )
    #fig_shap.subplots_adjust(top=0.86)
    plt.tight_layout(rect=[0, 0,1, 0.98])
    print("Saving figures...")
    fig_shap.savefig('Figs/Pre2020_Passed_Shap_Distribution.png', dpi=300, bbox_inches='tight')

    plt.close('all')
    print('Done')


if __name__ == '__main__':
    main()