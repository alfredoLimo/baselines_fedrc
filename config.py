# Overall settings
k_folds = 2 # number of folds for cross-validation, if 1, no cross-validation
strategy = 'FedEM'  # ['fedrc','FedEM','FeSEM','FedAvg']
random_seed = 42
n_clients = 5   
K_model = 3        # TODO, same as ifca
gpu = -1 # -1 for default best choice, [0,3] for cuda device
name_gpu = 0 # set the number  of the gpu to be used

# Dataset settings
dataset_name = "MNIST" # ["CIFAR10", "CIFAR100", "MNIST", "FMNIST", "EMNIST"]
drifting_type = 'static' # ['static', 'trND_teDR', 'trDA_teDR', 'trDA_teND', 'trDR_teDR', 'trDR_teND'] refer to ANDA page for more details
non_iid_type = 'feature_skew_strict' # refer to ANDA page for more details
verbose = True
count_labels = True
plot_clients = False
# careful with the args applying to your settings above
args = {
    'set_rotation': True,
    'set_color': True,
    'rotations':3,
    'colors':2,
    # 'py_bank': 5,
    # 'client_n_class': 5,
    # 'scaling_rotation_low':0.0,
    # 'scaling_rotation_high':0.0,
    # 'scaling_color_low':0.0,
    # 'scaling_color_high':0.0,
    # 'random_order':True
    # 'random_mode':2,
}

# Training model settings
model_name = "LeNet5"  # DEFAULT LeNet5 # ["LeNet5", "ResNet9"]
batch_size = 64
test_batch_size = 64
client_eval_ratio = 0.2
n_rounds = 5  # you need at least 4 rounds to start clustering
local_epochs = 2
lr = 0.005
momentum = 0.9


# self-defined settings
# n_classes_dict = {
#     "CIFAR10": 10,
#     "CIFAR100": 100,
#     "MNIST": 10,
#     "FMNIST": 10
# }
# n_classes = n_classes_dict[dataset_name]

# input_size_dict = {
#     "CIFAR10": (32, 32),
#     "CIFAR100": (32, 32),
#     "MNIST": (28, 28),
#     "FMNIST": (28, 28)
# }
# input_size = input_size_dict[dataset_name]

# acceptable_accuracy = {
#     "CIFAR10": 0.5,
#     "CIFAR100": 0.1,
#     "MNIST": 0.8,
#     "FMNIST": 0.8
# }
# th_accuracy = acceptable_accuracy[dataset_name]
training_drifting = False if drifting_type in ['static', 'trND_teDR'] else True # to be identified
default_path = f"{random_seed}/{model_name}/{dataset_name}/{drifting_type}"

# FL settings - Communications
port = '8098'
ip = '0.0.0.0' # Local Host=0.0.0.0, or IP address of the server
