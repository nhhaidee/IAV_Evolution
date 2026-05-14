import pandas as pd
import numpy as np
from codon_analyzer import CodonAnalyzer
import re
from utils import parse_vadr_cds_tbl, format_cds_coordinates
import glob
import pyfastx
from collections import defaultdict
from adjustText import adjust_text
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    segments = ['01_PB2', '02_PB1', '03_PA', '04_HA', '05_NP', '06_NA', '07_MP', '08_NS']
    product = ['polymerase PB2', 'polymerase PB1', 'polymerase PA', 'hemagglutinin', 'nucleocapsid protein', 'neuraminidase', 'matrix protein 1', 'nonstructural protein 1']

    for product_id, segment in enumerate(segments):

        print(f"Processing segment: {segment}")
        data_path = f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/data/{segment}'

        X_test = np.load(f'/{data_path}/X_test_onehot.npy')
        y_test = np.load(f'/{data_path}/y_test.npy')
        

        vadr_path = f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/VADR/VADR/{segment}_VADR'
        cds_tbl_path = f'{vadr_path}/{segment}_VADR.vadr.pass.tbl'
        parsed_cds_tbl_data = parse_vadr_cds_tbl(cds_tbl_path)
        cds_coords_dict = format_cds_coordinates(parsed_cds_tbl_data)

        segment_fa = pyfastx.Fasta(f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/VADR/Sequences/{segment}.fa', build_index=True)

        for fold in range(10):
            print(f"Processing fold: {fold}")

            # pred will be changed for each fold
            y_pred_data_path = f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/models_weight/{segment}/Baseline_preds_cal_fold_{fold}.npy'
            y_pred = np.load(y_pred_data_path)
            print(f"Loaded {y_pred_data_path} with shape: {y_pred.shape}")

            metadata_path = f'/fs/vnas_Hcfia/orph/hon000/part2/metadata/{segment}/Part1_2_{segment}_Test_Filtered.csv'
            df_test_metadata = pd.read_csv(metadata_path)
            df_test_metadata = df_test_metadata.reset_index(drop=True)
            df_test_metadata['y_true'] = y_test
            df_test_metadata['y_pred'] = y_pred
            print(f"Loaded test_metadata {metadata_path} for fold {fold} with shape: {df_test_metadata.shape}")

            shap_files = []
            shap_path = f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/shap/{segment}'
            for chunk in range(20):
                shap_files.append(f'{shap_path}/{segment}_shap_values_Fold_{fold}_chunk_{chunk}.npy')
            shap_values = np.concatenate([np.load(f) for f in shap_files])
            
            print ("shap values shape:", shap_values.shape)

            codon_analyzer = CodonAnalyzer()
            df_header = ['Seq_ID', 'True_Label', 'Pred_Label' , 'Clade', 'Subtype']
            df_codon_header = codon_analyzer.get_header()
            df_header.extend(df_codon_header)

            rscu_table = []
            shap_codon_table = []

            tmp_seg = segment.lstrip('0')
            convert_segment = tmp_seg + '_ID'

            for index, row in df_test_metadata.iterrows():
                #if row['y_true'] == row['y_pred']: # correct prediction only

                clade = row['Clade']
                subtype = row['Subtype']
                seq_id = row[convert_segment]
                sequence = segment_fa[seq_id].seq.upper()
                cds_coords = cds_coords_dict[seq_id][product[product_id]]
                extracted_coords = cds_coords['coordinate']
                codon_start = cds_coords.get('codon_start', 1)

                true_label = y_test[index] # get shap score for true label
                shap = shap_values[index, :, :, true_label] * X_test[index, :, :]
                #shap = shap_values[index, :, :, 1] * X_test[index, :, :] #get Avian class

                codon_counts = codon_analyzer.count_codons(seq_id, sequence, extracted_coords, codon_start)
                rscu_arr       = [seq_id, true_label, y_pred[index], clade, subtype]
                codon_shap_arr = [seq_id, true_label, y_pred[index], clade, subtype]

                rscu = codon_analyzer.calculate_rscu(codon_counts)
                rscu_arr.extend([codon_rscu for codon, codon_rscu in rscu.items()])
                assert df_codon_header == [codon for codon, codon_rscu in rscu.items()], "Codon headers do not match!"
                rscu_table.append(rscu_arr)

                codon_shap_scores, _ = codon_analyzer.calculate_codon_shap_scores(seq_id, sequence, extracted_coords, shap, codon_start)
                codon_shap_arr.extend([codon_shap for codon, codon_shap in codon_shap_scores.items()])
                assert df_codon_header == [codon for codon, codon_shap in codon_shap_scores.items()], "Codon headers do not match!"
                
                shap_codon_table.append(codon_shap_arr)
                

            df_rscu = pd.DataFrame(rscu_table, columns=df_header)
            df_shap = pd.DataFrame(shap_codon_table, columns=df_header)
            df_rscu.to_csv(f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/Codon_Usage_Shap/{segment}/All_Avian_Class_Post2020_{segment}_RSCU_Fold_{fold}.csv')
            df_shap.to_csv(f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/Codon_Usage_Shap/{segment}/All_Avian_Class_Post2020_{segment}_SHAP_Fold_{fold}.csv')

if __name__ == '__main__':
    main()
