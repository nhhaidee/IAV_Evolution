import numpy as np
import torch
import random
from utils import get_waveSeekernet_model
from sampling import resampling
from sklearn.model_selection import RepeatedStratifiedKFold
from shap import DeepExplainer, GradientExplainer
import logging
from time import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def set_seed(random_seed):
    print("Set Global Seed\n")
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def main():
    set_seed(0)#0 before
    segment = "07_MP"
    
    data_path = f"/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/data/{segment}"
    weight_path = f"/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/models_weight/{segment}"
    shap_out = f"/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/shap/H5Nx_North_America/{segment}"
    
    unfold_size = 12

    X_train = np.load(f'{data_path}/X_train_onehot.npy')
    y_train = np.load(f'{data_path}/y_train.npy')
    X_test  = np.load(f'/gpfs/fs7/grdi/genarcc/wp1/genomics_unit/IAV/part2/new/data/H5Nx_North_America/{segment}/X_test_onehot.npy')

    X_test_chunks = np.array_split(X_test, 10)
    del X_test
    splitter = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=0)
    for fold in range(10):

        train_indices, _ = list(splitter.split(X_train, y_train))[fold]
        X_CV_train, y_CV_train = resampling(X_train[train_indices], y_train[train_indices], n_downsamples=16000, n_upsamples=600,seed=0)

        logging.info("Resampled Train shape: %s, %s", X_CV_train.shape, y_CV_train.shape)
        logging.info("Class distribution:\n%s", np.transpose(np.unique(y_CV_train, return_counts=True)))

        human_index = np.where(y_CV_train == 0, True, False)
        avian_index = np.where(y_CV_train == 1, True, False)
        mammal_index = np.where(y_CV_train == 2, True, False)

        X_CV_train_human = X_CV_train[human_index]
        X_CV_train_avian = X_CV_train[avian_index]
        X_CV_train_mammal = X_CV_train[mammal_index]

        background_human = X_CV_train_human[np.random.choice(X_CV_train_human.shape[0], 350, replace=False)]
        background_avian = X_CV_train_avian[np.random.choice(X_CV_train_avian.shape[0], 350, replace=False)]
        background_mammal = X_CV_train_mammal[np.random.choice(X_CV_train_mammal.shape[0], 300, replace=False)]

        background_concat = np.concatenate((background_human, background_avian, background_mammal), axis=0)
        logging.info("Background data shape: %s", background_concat.shape)

        del background_human, background_avian, background_mammal, X_CV_train, y_CV_train, X_CV_train_mammal

        background_data = torch.tensor(background_concat, device='cuda:0')

        waveseekernet_weight = f"{weight_path}/Ablation_weight_{fold}_Baseline.pt"
        logging.info("Loading model weight: %s", waveseekernet_weight)

        wave_model = get_waveSeekernet_model(seq_len=background_concat.shape[2], res_len=5, unfold_size=unfold_size, model_weight=waveseekernet_weight)
        wave_model.eval()
        explainer = GradientExplainer(wave_model, background_data, batch_size=256)

        for chunk_id in range(10):
            start = time()
            logging.info("Explaining Chunk %d with shape %s", chunk_id, X_test_chunks[chunk_id].shape)
            X_expl = torch.tensor(X_test_chunks[chunk_id], device="cuda:0")
            raw_explanations = explainer.shap_values(X_expl)
            end = time()
            total_time = end - start
            logging.info("Done %d in %f seconds", chunk_id, total_time)
            np.save(f"{shap_out}/{segment}_shap_values_Fold_{fold}_chunk_{chunk_id}.npy", raw_explanations)
            del raw_explanations


if __name__ == '__main__':
    main()
