# ANDA interface

import config as cfg

import numpy as np
import argparse
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from ANDA import anda



# Get arguments
parser = argparse.ArgumentParser(description='Generate datasets for ANDA')
parser.add_argument('--fold', type=int, default=0, help='Fold number of the cross-validation')
parser.add_argument('--scaling', type=int, default=0, help='scaling num')
args = parser.parse_args()
 
 
 
cur_args = {}
if cfg.non_iid_type == 'feature_skew_strict':
    #1-4
    if args.scaling == 1:
        cur_rot = 2
        cur_col = 1
    elif args.scaling == 2:
        cur_rot = 2
        cur_col = 3
    elif args.scaling == 3:
        cur_rot = 4
        cur_col = 1
    elif args.scaling == 4:
        cur_rot = 4
        cur_col = 3
    else:
        raise KeyError
    
    # cur_rot = (args.scaling + 1) // 2
    # cur_col = 1 if args.scaling % 2 != 0 else 3
 
    print(f"Rotation: {cur_rot}, Color: {cur_col}")
    cur_args = {
        'set_rotation': True,
        'set_color': True,
        'rotations':cur_rot,
        'colors':cur_col,
    }
    
elif cfg.non_iid_type == 'label_skew_strict':
    #1-4
    if args.scaling == 1:
        cur_class  = 8
    elif args.scaling == 2:
        cur_class  = 6
    elif args.scaling == 3:
        cur_class  = 4
    elif args.scaling == 4:
        cur_class  = 2
    else:
        raise KeyError
    
    # cur_class = 10-args.scaling
    print(f"Class: {cur_class}")
 
    cur_args = {
        'client_n_class':cur_class,
        'py_bank':5,
    }
    
elif cfg.non_iid_type == 'feature_condition_skew':
    #1-4    
    # cur_scaling = args.scaling /10
    cur_args = {
        'random_mode':True,
        'mixing_label_number':args.scaling+2,
        'scaling_label_low':1,
        'scaling_label_high':1,
    }
    
elif cfg.non_iid_type == 'label_condition_skew':
    #1-4
    if args.scaling == 1:
        cur_class  = 2
    elif args.scaling == 2:
        cur_class  = 4
    elif args.scaling == 3:
        cur_class  = 6
    elif args.scaling == 4:
        cur_class  = 8
    else:
        raise KeyError
    
    # cur_class = args.scaling +1
    print(f"Class: {cur_class}")
 
    cur_args = {
        'set_rotation': True,
        'set_color': True,
        'rotations':4,
        'colors':1,
        'random_mode':True,
        'rotated_label_number':cur_class,
        'colored_label_number':cur_class,
    }    
else:
    raise ValueError("Non-IID type not found! Please check the ANDA page for more details.")
 





# valid dataset names
assert cfg.dataset_name in ['CIFAR10', 'CIFAR100', 'MNIST', 'FMNIST', 'EMNIST'], \
        "Dataset not found! Please check the ANDA page for more details."

# Create folder if not exist
os.makedirs('./data/cur_datasets', exist_ok=True)

anda_dataset = []

# special static mode using unique fn
if cfg.drifting_type == 'static':
    assert cfg.non_iid_type in ['feature_skew',
                                'label_skew',
                                'feature_condition_skew',
                                'label_condition_skew',
                                'split_unbalanced',
                                'feature_label_skew',
                                'feature_condition_skew_with_label_skew',
                                'label_condition_skew_with_label_skew',
                                'label_condition_skew_unbalanced',
                                'feature_condition_skew_unbalanced',
                                'label_skew_unbalanced',
                                'feature_skew_unbalanced',
                                'feature_skew_strict',
                                'label_skew_strict',
                                'feature_condition_skew_strict',
                                'label_condition_skew_strict',
    ], "Non-IID type not supported in static mode! Please check the ANDA page for more details."
    anda_dataset = anda.load_split_datasets(
        dataset_name = cfg.dataset_name,
        client_number = cfg.n_clients,
        non_iid_type = cfg.non_iid_type,
        mode = "manual",
        verbose = cfg.verbose,
        count_labels=cfg.count_labels,
        plot_clients=cfg.plot_clients,
        random_seed = cfg.random_seed + args.fold,
        **cur_args
    )
elif cfg.drifting_type in ['trND_teDR','trDA_teDR','trDA_teND','trDR_teDR','trDR_teND']:
    # dynamic mode using same fn
    anda_dataset = anda.load_split_datasets_dynamic(
        dataset_name = cfg.dataset_name,
        client_number = cfg.n_clients,
        non_iid_type = cfg.non_iid_type,
        drfting_type = cfg.drifting_type,
        verbose = cfg.verbose,
        count_labels=cfg.count_labels,
        plot_clients=cfg.plot_clients,
        random_seed = cfg.random_seed + args.fold,
        **cur_args
    )
else:
    raise ValueError("Drifting type not found! Please check the ANDA page for more details.")

# Save anda_dataset
# simple format as training not drifting
n_clusters = []
if not cfg.training_drifting:
    for client_number in range(cfg.n_clients):
        np.save(f'./data/cur_datasets/client_{client_number}', anda_dataset[client_number])
        print(f"Data for client {client_number} saved")
        # print(anda_dataset[client_number]["cluster"])
        n_clusters.append(anda_dataset[client_number]["cluster"])
    n_clusters = np.unique(n_clusters).shape[0]
    print(f"Number of clusters: {n_clusters}")
    np.save(f'./data/cur_datasets/n_clusters.npy', n_clusters)

# complex format as training drifting
else:
    drifting_log = {}
    for dataset in anda_dataset:
        client_number = dataset['client_number']
        cur_drifting_round = int(cfg.n_rounds * dataset['epoch_locker_indicator']) if dataset['epoch_locker_indicator'] != -1 else -1

        # save data file      
        filename = f'./data/cur_datasets/client_{client_number}_round_{cur_drifting_round}.npy'
        np.save(filename, dataset)
        print(f"Data for client {client_number} round {cur_drifting_round} saved")

        # log drifting round info
        if client_number not in drifting_log:
            drifting_log[client_number] = []
        drifting_log[client_number].append(cur_drifting_round)

    # print(", ".join(f"{key}: {value}" for key, value in drifting_log.items()))

    # save log file
    np.save(f'./data/cur_datasets/drifting_log.npy', drifting_log)

print("Datasets saved successfully!")

