import random
import torch
import numpy as np
from config import get_default_config
import sklearn
from dataset import build_dataloader
from train import run_training, run_predicting
from optimizer import get_optimizer
from model.compose_model import ComposeModel
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

def set_all_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    sklearn.random.seed(seed)

if __name__ == '__main__':

    config = get_default_config()
    set_all_seed(config.seed)


    df = pd.read_csv('./data/preprocessed_cohort_2.csv')
    str_kfold = StratifiedKFold(n_splits=config.kfold, shuffle=True, random_state=config.seed)

    train_history = []; 

    for train_index, test_index in str_kfold.split(df, df.label):

        all_df = df.iloc[train_index].copy()
        test_df = df.iloc[test_index].copy()
        for train_index, valid_index in str_kfold.split(all_df, all_df.label):
            train_df = all_df.iloc[train_index].copy()
            valid_df = all_df.iloc[valid_index].copy()
            break
        del all_df

        scaler = RobustScaler()
        train_dl = build_dataloader(config=config, df=train_df, scaler=scaler, is_train=True, batch_size=config.batch_size, shuffle=True)
        valid_dl = build_dataloader(config=config, df=valid_df, scaler=scaler, is_train=False, batch_size=config.batch_size, shuffle=True)
        test_dl = build_dataloader(config=config, df=test_df, scaler=scaler, is_train=False, batch_size=config.batch_size, shuffle=False)


        device = torch.device(config.device)
        model = ComposeModel(config).to(device)
        optimizer = get_optimizer('adam', config, model.parameters())
        
        history = run_training(config, model, train_dl, valid_dl, optimizer, device)
        test_metric_scores, test_y_probs = run_predicting(config, model, test_dl, device)
        print(history)
        print(test_metric_scores)
        print(test_y_probs)
        
        