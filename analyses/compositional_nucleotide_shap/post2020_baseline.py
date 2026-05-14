import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold
from util import get_shap_composition_vectorized, shap_distribution_plot, nt_map
from sklearn.metrics import classification_report
import matplotlib.patches as mpatches


def main():
    segments = ['01_PB2', '02_PB1', '03_PA', '04_HA', '05_NP', '06_NA', '07_MP', '08_NS']
    n_rows = 3 * 2
    n_cols = int(len(segments) /2)

    fig_shap, axes_shap = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows), sharey=False)
    #fig_shap.suptitle('Nucleotide SHAP Score Distribution (Pre2020)', fontsize=24)

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
    
        print(f'Processing segment {segment} for column {col_idx+1}/{n_cols}...')

        test_data_path = f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/data/{segment}'
        shap_data_path = f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/shap/{segment}'
        pred_data_path = f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/models_weight/{segment}'

        print(f'Test data path: {test_data_path}')
        print(f'Shap data path: {shap_data_path}')        
        print(f'Pred data path: {pred_data_path}')

        X_test = np.load(f'{test_data_path}/X_test_onehot.npy')
        y_test = np.load(f'{test_data_path}/y_test.npy')
        
        
        # Storage dictionary
        agg_data = {
            'Human': {'raw_sum': {k:[] for k in list(nt_map.values()) + ['Pad']}},
            'Avian': {'raw_sum': {k:[] for k in list(nt_map.values()) + ['Pad']}},
            'Non-human Mammal': {'raw_sum': {k:[] for k in list(nt_map.values()) + ['Pad']}}
        }
        species_idx = {'Human': 0, 'Avian': 1, 'Non-human Mammal': 2}

        for fold in range(10):
            print(f'Fold {fold}...')

            y_pred = np.load(f'{pred_data_path}/Baseline_preds_cal_fold_{fold}.npy')
            
            shap_files = []
            for chunk in range(20):
                shap_files.append(f'{shap_data_path}/{segment}_shap_values_Fold_{fold}_chunk_{chunk}.npy')
            
            shap_score = np.concatenate([np.load(f) for f in shap_files])

            print("X_test, y_test, y_pred shape:",X_test.shape, y_test.shape, y_pred.shape)
            print("y_test distribution:", np.unique(y_test, return_counts=True))
            print("Shap scores shape:",shap_score.shape)
            print("Classification report:")
            print(classification_report(y_test, y_pred, zero_division=np.nan))
            
            
            indices = {
                'Human': (y_pred == 0) & (y_pred == y_test),
                'Avian': (y_pred == 1) & (y_pred == y_test),
                'Non-human Mammal': (y_pred == 2) & (y_pred == y_test)
            }
            
            for species, mask in indices.items():
                if np.sum(mask) == 0: continue
                
                sp_onehot = X_test[mask]
                sp_shap = shap_score[mask]
                
                comp = get_shap_composition_vectorized(sp_onehot, sp_shap, species_idx[species])
                # Aggregate Raw Sums
                for nt in nt_map.values():
                    agg_data[species]['raw_sum'][nt].extend(comp['raw_sum'][nt])
                agg_data[species]['raw_sum']['Pad'].extend(comp['pad_sum'])

            del shap_score # Free memory    

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
    plt.tight_layout(rect=[0, 0,1, 0.98])
       
    shap_plot_file = 'Figs/Post2020_Passed_Shap_Distribution.png'
    fig_shap.savefig(shap_plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig_shap) 
    print(f'\nDone')


if __name__ == '__main__':
    main()