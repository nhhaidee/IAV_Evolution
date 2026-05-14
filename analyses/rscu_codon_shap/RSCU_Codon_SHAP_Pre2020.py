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
from sklearn.model_selection import RepeatedStratifiedKFold
import seaborn as sns

def main():

    segments = ['01_PB2', '02_PB1', '03_PA', '04_HA', '05_NP', '06_NA', '07_MP', '08_NS']
    product = ['polymerase PB2', 'polymerase PB1', 'polymerase PA', 'hemagglutinin', 'nucleocapsid protein', 'neuraminidase', 'matrix protein 1', 'nonstructural protein 1']
    

    for product_id, segment in enumerate(segments):

        print(f"Processing segment: {segment}")
        data_path = f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/data/{segment}'
        
        X_train = np.load(f'{data_path}/X_train_onehot.npy')
        y_train = np.load(f'{data_path}/y_train.npy')   

        vadr_path = f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/VADR/VADR/{segment}_VADR'
        cds_tbl_path = f'{vadr_path}/{segment}_VADR.vadr.pass.tbl'
        parsed_cds_tbl_data = parse_vadr_cds_tbl(cds_tbl_path)
        cds_coords_dict = format_cds_coordinates(parsed_cds_tbl_data)

        segment_fa = pyfastx.Fasta(f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/VADR/Sequences/{segment}.fa', build_index=True)

        n_seq      = X_train.shape[0]
        res_len    = X_train.shape[1]
        seq_len    = X_train.shape[2]
        num_labels = len(np.unique(y_train))
        shap_oof = np.zeros((n_seq, res_len, seq_len, num_labels), dtype=np.float32)
        n_chunks = 5
        splitter = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=0)
        
        for fold in range(10):
        
            print(f"Processing fold: {fold}")
            _, val_indices = list(splitter.split(X_train, y_train))[fold]
            shap_files = []
            shap_path = f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/shap/{segment}'
            for chunk in range(n_chunks):
                shap_files.append(f'{shap_path}/{segment}_training_shap_values_Fold_{fold}_chunk_{chunk}.npy')
            shap_values = np.concatenate([np.load(f) for f in shap_files])
            print ("shap values shape:", shap_values.shape)
            shap_oof[val_indices,...] = shap_values
        if segment == '01_PB2':
            np.save(f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/Codon_Usage_Shap/{segment}/{segment}_Training_OOF.npy', shap_oof)
        print ("shap oof shape:", shap_oof.shape)
        metadata_path = f'/fs/vnas_Hcfia/orph/hon000/part2/metadata/{segment}/{segment}_Train_Filtered.csv'
        df_test_metadata = pd.read_csv(metadata_path)
        df_test_metadata = df_test_metadata.reset_index(drop=True)
        print(f"Loaded test_metadata {metadata_path} with shape: {df_test_metadata.shape}")
        codon_analyzer = CodonAnalyzer()
        df_header = ['Seq_ID', 'True_Label', 'Clade', 'Subtype']
        df_codon_header = codon_analyzer.get_header()
        df_header.extend(df_codon_header)

        rscu_table = []
        shap_codon_table = []

        tmp_seg = segment.lstrip('0')
        convert_segment = tmp_seg + '_ID'

        for index, row in df_test_metadata.iterrows():

            clade = row['Clade']
            subtype = row['Subtype']
            host_id = row['Host_ID'] # true label
            seq_id = row[convert_segment]
            sequence = segment_fa[seq_id].seq.upper()
            if 'neuraminidase-like protein' in cds_coords_dict[seq_id]:
                print (seq_id, cds_coords_dict[seq_id])
                cds_coords = cds_coords_dict[seq_id]['neuraminidase-like protein']
            else:
                cds_coords = cds_coords_dict[seq_id][product[product_id]]
            extracted_coords = cds_coords['coordinate']
            codon_start = cds_coords.get('codon_start', 1)

            shap = shap_oof[index, :, :, host_id] * X_train[index, :, :]
            #shap = shap_oof[index, :, :, 1] * X_train[index, :, :] #get avian

            codon_counts = codon_analyzer.count_codons(seq_id, sequence, extracted_coords, codon_start)
            rscu_arr       = [seq_id, host_id, clade, subtype]
            codon_shap_arr = [seq_id, host_id, clade, subtype]

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
        df_rscu.to_csv(f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/Codon_Usage_Shap/{segment}/All_Avian_Class_Training_{segment}_RSCU.csv')
        df_shap.to_csv(f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/Codon_Usage_Shap/{segment}/All_Avian_Class_Training_{segment}_SHAP.csv')
            
            


if __name__ == '__main__':
    main()
