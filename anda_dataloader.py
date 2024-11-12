from ANDA import anda
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

import config as cfg

class ClientDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.targets = self.labels  
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index], index

def get_ANDA_loaders():
    train_iterators, val_iterators, test_iterators, client_types, client_features = [], [], [], [], []
    for client_number in range(cfg.n_clients):
        client_data = np.load(f'./data/cur_datasets/client_{client_number}.npy', allow_pickle=True).item()

        cur_features_train_all = client_data['train_features']
        cur_labels_train_all = client_data['train_labels']
        train_features, val_features, train_labels, val_labels = train_test_split(
            cur_features_train_all, cur_labels_train_all, test_size=cfg.client_eval_ratio, random_state=cfg.random_seed
        )
        test_features = client_data['test_features']
        test_labels = client_data['test_labels']

        val_features = torch.cat([torch.tensor(val_features), torch.tensor(val_features), torch.tensor(val_features), torch.tensor(val_features)], dim=0)
        val_labels = torch.cat([torch.tensor(val_labels), torch.tensor(val_labels), torch.tensor(val_labels), torch.tensor(val_labels)], dim=0)
        
        # reduce client data
        if cfg.n_samples_clients > 0:
            train_features = train_features[:cfg.n_samples_clients]
            train_labels = train_labels[:cfg.n_samples_clients]
            val_features = val_features[:cfg.n_samples_clients]
            val_labels = val_labels[:cfg.n_samples_clients]
            
        if cfg.non_iid_type in ['feature_condition_skew','label_skew_strict','feature_condition_skew_strict']:
            train_features = train_features.unsqueeze(1)
            val_features = val_features.unsqueeze(1)
            test_features = test_features.unsqueeze(1)
 

        train_data = ClientDataset(train_features, train_labels)
        val_data = ClientDataset(val_features, val_labels)
        test_data = ClientDataset(test_features, test_labels)

        train_iterators.append(torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True))
        val_iterators.append(torch.utils.data.DataLoader(val_data, batch_size=cfg.test_batch_size, shuffle=False))
        test_iterators.append(torch.utils.data.DataLoader(test_data, batch_size=cfg.test_batch_size, shuffle=False))

        client_types.append(None)
        client_features.append([0] * 10)

    return train_iterators, val_iterators, test_iterators, client_types, client_features
