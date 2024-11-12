"""Run Experiment

This script allows to run one federated learning experiment; the experiment name, the method and the
number of clients/tasks should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * run_experiment - runs one experiments given its arguments
"""
from sklearn import cluster
from utils.utils import *
from utils.constants import *
from utils.args import *
from anda_dataloader import get_ANDA_loaders

from torch.utils.tensorboard import SummaryWriter

import config

def init_clients(args_, root_path, logs_dir):
    """
    initialize clients from data folders
    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")
    class_number = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]
            
    # new
    if config.dataset_name in ['MNIST', 'FMNIST', 'CIFAR10', 'CIFAR100']:
        train_iterators, val_iterators, test_iterators, client_types, feature_types =\
            get_ANDA_loaders()

    elif LOADER_TYPE[args_.experiment] == 'cifar10-c':
        if 'test' in root_path:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation,
                    test = True,
                    test_num = 3
                )
        else:
            train_iterators, val_iterators, test_iterators, client_types, feature_types =\
                get_cifar10C_loaders(
                    root_path='./data/cifar10-c',
                    batch_size=args_.bz,
                    is_validation=args_.validation
                )

    else:
        raise ValueError("Unknown dataset: {}".format(args_.experiment))

    # return all datasets

    print("===> Initializing clients..")
    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):

        # if task_id == 0:
        if False:
            print(len(train_iterator.dataset))
            print(len(val_iterator.dataset))
            # Inspect train_iterator
            train_batch = next(iter(train_iterator))
            if isinstance(train_batch, (list, tuple)):
                if len(train_batch) == 2:
                    train_inputs, train_targets = train_batch
                elif len(train_batch) == 3:
                    train_inputs, train_targets, _ = train_batch
                else:
                    print("Unexpected train batch structure:", train_batch)
            print(f"Task {task_id} (Train): Inputs shape: {train_inputs.shape}, Targets shape: {train_targets.shape}")

            # Inspect val_iterator
            val_batch = next(iter(val_iterator))
            if isinstance(val_batch, (list, tuple)):
                if len(val_batch) == 2:
                    val_inputs, val_targets = val_batch
                elif len(val_batch) == 3:
                    val_inputs, val_targets, _ = val_batch
                else:
                    print("Unexpected val batch structure:", val_batch)
            print(f"Task {task_id} (Validation): Inputs shape: {val_inputs.shape}, Targets shape: {val_targets.shape}")

            # Inspect test_iterator
            test_batch = next(iter(test_iterator))
            if isinstance(test_batch, (list, tuple)):
                if len(test_batch) == 2:
                    test_inputs, test_targets = test_batch
                elif len(test_batch) == 3:
                    test_inputs, test_targets, _ = test_batch
                else:
                    print("Unexpected test batch structure:", test_batch)
            print(f"Task {task_id} (Test): Inputs shape: {test_inputs.shape}, Targets shape: {test_targets.shape}")

        if train_iterator is None or test_iterator is None:
            continue

        learners_ensemble =\
            get_learners_ensemble(
                n_learners=args_.n_learners,
                client_type=CLIENT_TYPE[args_.method],
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                n_gmm=args_.n_gmm,
                embedding_dim=args_.embedding_dimension,
                hard_cluster=args_.hard_cluster,
                binary=args_.binary,
                phi_model=args.phi_model
            )

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        client = get_client(
            client_type=CLIENT_TYPE[args_.method],
            learners_ensemble=learners_ensemble,
            q=args_.q,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=args_.local_steps,
            tune_locally=args_.locally_tune_clients,
            data_type = client_types[task_id],
            feature_type = feature_types[task_id],
            class_number = class_number,
            id=task_id,
        )

        clients_.append(client)

    return clients_


def run_experiment(args_):
    torch.manual_seed(args_.seed)

    data_dir = get_data_dir(args_.experiment)

    if "logs_dir" in args_:
        logs_dir = args_.logs_dir
    else:
        logs_dir = os.path.join("logs", args_to_string(args_))

    print("==> Clients initialization..")
    clients = init_clients(args_, root_path=os.path.join(data_dir, "train"), logs_dir=os.path.join(logs_dir, "train"))

    # No test clients
    # print("==> Test Clients initialization..")
    # test_clients = init_clients(args_, root_path=os.path.join(data_dir, "test"),
    #                             logs_dir=os.path.join(logs_dir, "test"))
    test_clients = []

    # return

    logs_path = os.path.join(logs_dir, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_dir, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    global_learners_ensemble = \
        get_learners_ensemble(
            n_learners=args_.n_learners,
            client_type=CLIENT_TYPE[args_.method],
            name=args_.experiment,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu,
            embedding_dim=args_.embedding_dimension,
            n_gmm=args_.n_gmm,
            hard_cluster=args_.hard_cluster,
            binary=args_.binary,
            phi_model=args.phi_model
        )

    aggregator_type = AGGREGATOR_TYPE[args_.method]

    aggregator =\
        get_aggregator(
            aggregator_type=aggregator_type,
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            lr_lambda=args_.lr_lambda,
            lr=args_.lr,
            q=args_.q,
            mu=args_.mu,
            communication_probability=args_.communication_probability,
            sampling_rate=args_.sampling_rate,
            log_freq=args_.log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=args_.verbose,
            seed=args_.seed,
            experiment = args_.experiment,
            method = args_.method,
            suffix = args_.suffix,
            split = args_.split,
            domain_disc=args_.domain_disc,
            em_step=args_.em_step
        )

    print("Training..")
    pbar = tqdm(total=args_.n_rounds)
    current_round = 0
    pre_action = 0
    mean_Is_pre = []
    rho = 0.3
    if_sufficient = False
    while current_round < args_.n_rounds:

        # if pre_action == 0:
        #     aggregator.mix(diverse=False)
        # else:
        #     aggregator.mix(diverse=False)
        aggregator.mix(diverse=False)


        C = CLASS_NUMBER[LOADER_TYPE[args_.experiment]]
        n_learner = aggregator.n_learners
        cluster_label_weights = [[0] * C for _ in range(n_learner)]
        cluster_weights = [0 for _ in range(n_learner)]
        global_flags = [[] for _ in range(n_learner)]
        
        with open('./logs/{}/sample-weight-{}-{}.txt'.format(args_.experiment, args_.method, args_.suffix), 'w') as f:
            for client_index, client in enumerate(clients):
                for i in range(len(client.train_iterator.dataset.targets)):
                    # if args_.method == 'FedSoft':
                    #     f.write('{},{},{}, {}\n'.format(client.data_type, client.train_iterator.dataset.targets[i], client.feature_types[i], aggregator.clusters_weights[client_index]))
                    # else:
                    #     f.write('{},{},{}, {}\n'.format(client.data_type, client.train_iterator.dataset.targets[i], client.feature_types[i], client.samples_weights.T[i]))
                    
                    for j in range(len(cluster_label_weights)):
                        cluster_weights[j] += client.samples_weights[j][i]
                f.write('\n')
        # else:
        #     for client_index, client in enumerate(clients):
        #         for i in range(len(client.train_iterator.dataset.targets)):
        #             for j in range(len(cluster_label_weights)):
        #                     cluster_weights[j] += client.samples_weights[j][i]

        # with open('./logs/{}/mean-I-{}-{}-{}.txt'.format(args_.experiment, args_.method, args_.gamma, args_.suffix), 'a+') as f:
        #     mean_Is = torch.zeros((len(clients),))
        #     clusters = torch.zeros((len(clients),))
        #     client_types = torch.zeros((len(clients),))
        #     for i, client in enumerate(clients):
        #         mean_Is[i] = client.mean_I
        #         client_types[i] = client.data_type
        #         # clusters[i] = torch.nonzero(client.cluster==torch.max(client.cluster)).squeeze()
        #     f.write('{}'.format(mean_Is))
        #     f.write('\n')
        # with open('./logs/{}/cluster-weights-{}-{}-{}.txt'.format(args_.experiment, args_.method, args_.gamma, args_.suffix), 'a+') as f:
        #     f.write('{}'.format(cluster_weights))
        #     f.write('\n')
#        print(cluster_weights) ?
        # print(client_types)
        # print(clusters)



        # K = 0
        # for i in range(n_learner):
        #     if n_learner == 1:
        #         break
        #     if cluster_weights[i] <= sum(cluster_weights) * args_.gamma:
        #         # print(i)
        #         for client in clients:
        #             client.remove_learner(i - K)
        #         for client in test_clients:
        #             client.remove_learner(i - K)
        #         aggregator.remove_learner(i - K)
        #         K += 1
        #         cluster_label_weights.pop(i - K)
        #         if_sufficient = True
        

        for client in clients:
            client_labels_learner_weights = client.labels_learner_weights
            for j in range(len(cluster_label_weights)):
                for k in range(C):
                    cluster_label_weights[j][k] += client_labels_learner_weights[j][k]
        for j in range(len(cluster_label_weights)):
            for i in range(len(cluster_label_weights[j])):
                if cluster_label_weights[j][i] < 1e-8:
                    cluster_label_weights[j][i] = 1e-8
            cluster_label_weights[j] = [i / sum(cluster_label_weights[j]) for i in cluster_label_weights[j]]


        for client in clients:
            client.update_labels_weights(cluster_label_weights)
        # for client in test_clients:
        #     client.update_labels_weights(cluster_label_weights)

        for client in test_clients:
            print(client.mean_I, client.cluster, torch.nonzero(client.cluster==torch.max(client.cluster)).squeeze())

        if aggregator.c_round != current_round:
            pbar.update(1)
            current_round = aggregator.c_round


    if "save_dir" in args_:
        save_dir = os.path.join(args_.save_dir)

        os.makedirs(save_dir, exist_ok=True)
        aggregator.save_state(save_dir)

def mean_results(n_clients, fold):
    x = []
    for i in range(n_clients):
        x.append(np.load(f"results_{i}.npy"))

    # delete files
    for i in range(n_clients):
        os.remove(f"results_{i}.npy")
        
    # delete previous average_cluster.pt file
    os.remove('average_cluster.pt')
        
    # stack
    x = np.stack(x, axis=0)   

    # Save metrics as numpy array
    metrics = {
        "loss": list(x[:,0]),
        "accuracy": list(x[:,1]),
        "average_loss": np.mean(x[:,0]),
        "average_accuracy": np.mean(x[:,1]),
        "loss_expected": list(x[:,2]),
        "accuracy_expected": list(x[:,3]),
        "average_loss_expected": np.mean(x[:,2]),
        "average_accuracy_expected": np.mean(x[:,3]),
        "loss_cluster": list(x[:,4]),
        "accuracy_cluster": list(x[:,5]),
        "average_loss_cluster": np.mean(x[:,4]),
        "average_accuracy_cluster": np.mean(x[:,5]), 
    }
    np.save(f'test_metrics_fold_{fold}.npy', metrics)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()

    # update the real values from config.py
    args.experiment = 'cifar10-c'   # DO NOT CHANGE
    # args.method = config.strategy           
    args.n_learners = np.load(f'./data/cur_datasets/n_clusters.npy').item()
    args.n_rounds = config.n_rounds
    args.bz = config.batch_size
    args.local_steps = config.local_epochs
    args.device = torch.device("cpu")
    args.lr = config.lr
    args.seed = config.random_seed + args.fold    
    
    if config.gpu == -1:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            args.device = torch.device("mps")
        elif torch.cuda.is_available():
            args.device = torch.device(f"cuda:{config.name_gpu}")
    else:
        args.device = torch.device("cuda:{}".format(config.gpu))

    print(f"USING DEVICE: {args.device}")

    run_experiment(args)
    
    # average results from all clients
    mean_results(config.n_clients, args.fold)
