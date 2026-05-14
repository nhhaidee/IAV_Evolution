import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score as ba_score
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, matthews_corrcoef
import random
from model import WaveSeekerClassifier
from sampling import get_rare_sequence, resampling
    
    
def set_seed(random_seed):
    print ("Set Global Seed\n")
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_score(y_true, y_pred):
    ba = ba_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    p_score = precision_score(y_true, y_pred, average="macro")
    r_score = recall_score(y_true, y_pred, average="macro")
    mcc = matthews_corrcoef(y_true, y_pred)
    print(ba, f1, p_score, r_score, mcc)
    print(classification_report(y_true, y_pred, zero_division=np.nan))
    return ba, f1, p_score, r_score, mcc


def main():

    seed=0
    set_seed(seed)

    train_path      = '/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/data/04_HA/'
    test_path       = '/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/data/04_HA/'   
    model_save_path = '/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/models_weight/04_HA/'
    ablation_result = '/fs/vnas_Hcfia/orph/hon000/part2/results/04_HA/'

    X_train = np.load(train_path + 'X_train_onehot.npy')
    y_train = np.load(train_path + 'y_train.npy')

    X_test_high_quality = np.load(test_path + 'X_test_onehot.npy')
    y_test_high_quality = np.load(test_path + 'y_test.npy')

    
    
    print("Train data shape:", X_train.shape, y_train.shape)
    print("Test High-quality Data Shape:", X_test_high_quality.shape, y_test_high_quality.shape)

    n_out = len(np.unique(y_train))
    seq_len = X_train.shape[2]
    res_len = 5
    patch_size = (11, res_len) # 15 before
    epochs = 35
    batch_size = 256
    emb_dim = 64
    final_hidden_size = 24
    n_splits = 10

    cv_cols = ["Model", "Balanced Accuracy", "F1-Score (Macro)", "Precision (Macro)", "Recall (Macro)", "MCC"]
    param_results_high_quality = []
    
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=0)
    for kfold_index, (train, test) in enumerate(splitter.split(X_train, y_train)):

        print("*************************Fold: ", kfold_index, "**********************************\n")

        X_CV_train, y_CV_train = resampling(X_train[train], y_train[train], n_downsamples=16000, n_upsamples=600,seed=seed) # up sampling human and avian, keep non-human mammals (set 600)

        X_CV_test  = X_train[test]
        y_CV_test  = y_train[test]
        
        print ("Train shape: ", X_CV_train.shape, y_CV_train.shape)
        print (np.transpose(np.unique(y_CV_train, return_counts=True)))
        print ("Val shape: ", X_CV_test.shape, y_CV_test.shape)
        print (np.transpose(np.unique(y_CV_test, return_counts=True)))

        
        clf = WaveSeekerClassifier(
            n_channels=1,
            seq_L=seq_len,
            res_L=res_len,
            patch_size=patch_size,
            n_out=n_out,
            batch_size=batch_size,
            emb_dim=emb_dim,
            final_hidden_size=final_hidden_size,
            epochs=epochs,
            patch_mode="patch",
            wavelet_names=["sym4"],
            n_blocks=1,
            lr=0.0025)

        model_name = "Baseline"
        model_weight = model_save_path + "Ablation_weight_" + str(kfold_index) + "_" + model_name + '.pt'

        clf.fit(X_CV_train, y_CV_train, X_CV_test, y_CV_test, save_path=model_weight)
        print("%s Result:" % model_name)

        print("High Quality (Post 2020)")
        ctest_high_quality = clf.predict(X_test_high_quality)
        np.save(model_save_path+model_name+"_high_quality_"+str(kfold_index)+".npy",ctest_high_quality)
        
        ba_main, f1_main, p_score_main, r_score_main, mcc_main = get_score(y_test_high_quality, ctest_high_quality)
        param_results_high_quality.append((model_name, ba_main, f1_main, p_score_main, r_score_main, mcc_main))

        del clf

        df_param_results_high_quality = pd.DataFrame(param_results_high_quality, columns=cv_cols)
        df_param_results_high_quality.to_csv(ablation_result + "Ablation_High_Quality_Fold_" + str(kfold_index) + ".csv")
        print ('\nHigh Quality Post 2020')
        print (df_param_results_high_quality.groupby(['Model'])['F1-Score (Macro)'].agg(['mean', 'std', 'sem', 'count']).to_string())
        print (df_param_results_high_quality.groupby(['Model'])['Balanced Accuracy'].agg(['mean', 'std', 'sem', 'count']).to_string())
        print (df_param_results_high_quality.groupby(['Model'])['MCC'].agg(['mean', 'std', 'sem', 'count']).to_string())


if __name__ == '__main__':
    main()
