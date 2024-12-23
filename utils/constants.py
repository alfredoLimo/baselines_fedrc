"""Configuration file for experiments"""
import string


LOADER_TYPE = {
    "synthetic": "tabular",
    "cifar10": "cifar10",
    "cifar10-c": "cifar10-c",
    "cifar10-c-imbalance": "cifar10-c-imbalance",
    "cifar10-c-concept-only": "cifar10-c-concept-only",
    "cifar10-c-concept-5": "cifar10-c-concept-5",
    "cifar10-c-concept-feature": "cifar10-c-concept-feature",
    "cifar10-c-concept-feature-label": "cifar10-c-concept-feature-label",
    "cifar10-c-concept-label": "cifar10-c-concept-label",
    "cifar10-c-feature-only": "cifar10-c-feature-only",
    "cifar10-c-label-only": "cifar10-c-label-only",
    "cifar10-c-2swap": "cifar10-c-2swap",
    "cifar10-c-4swap": "cifar10-c-4swap",
    "cifar100-c-4swap": "cifar100-c-4swap",
    "cifar100-c-2swap": "cifar100-c-2swap",
    "cifar10-c-noisy-02": "cifar10-c-noisy-02",
    "cifar10-c-noisy-01": "cifar10-c-noisy-01",
    "cifar10-c-noisy-type2-02": "cifar10-c-noisy-type2-02",
    "cifar10-c-noisy-type2-04": "cifar10-c-noisy-type2-04",
    "cifar10-c-noisy-04": "cifar10-c-noisy-04",
    "cifar100-c-noisy-02": "cifar100-c-noisy-02",
    "cifar100-c-noisy-type2-04": "cifar100-c-noisy-type2-04",
    "cifar100-c": "cifar100-c",
    "cifar100-c-10": "cifar100-c-10",
    "cifar100": "cifar100",
    "tiny-imagenet-c": "tiny-imagenet-c",
    "tiny-imagenet-c-2swap": "tiny-imagenet-c-2swap",
    "tiny-imagenet-c-4swap": "tiny-imagenet-c-4swap",
    "emnist": "emnist",
    "femnist": "femnist",
    "shakespeare": "shakespeare",
    "fmnist-c": "fmnist-c",
    'airline': 'airline',
    'powersupply': 'powersupply',
    'elec': 'elec',
    # new
    'MNIST': 'MNIST',
    'FMNIST': 'FMNIST',
    'CIFAR10': 'CIFAR10',
    'CIFAR100': 'CIFAR100'
}

EXTENSIONS = {
    "tabular": ".pkl",
    "cifar10": ".pkl",
    "cifar100": ".pkl",
    "emnist": ".pkl",
    "femnist": ".pt",
    "shakespeare": ".txt",
}

AGGREGATOR_TYPE = {
    "FedEM": "centralized",         # tag
    "FedEM_SW": "FedIAS",
    "fedrc": "centralized",         # tag
    "fedrc_tune": "centralized",
    "fedrc_Adam": "centralized",
    "fedrc_DP": "centralized",
    "fedrc_SW": "FedIAS",
    "stoCFL": "STOCFLAggregator",
    "ICFL": "ICFLAggregator",
    "FedAvg": "centralized",
    "FedProx": "centralized",
    "local": "no_communication",
    "pFedMe": "personalized",
    "clustered": "clustered",
    "APFL": "APFL",
    "L2SGD": "L2SGD",
    "AFL": "AFL",
    "FFL": "FFL",
    "IFCA": "IFCA",
    "FeSEM": "FeSEM",           # tag
    "FeSEM_SW": "FedIAS",
    "FedSoft": "FedSoft",
    "FedGMM": "ACGcentralized"
}

CLIENT_TYPE = {
    "FedEM": "mixture",         # tag   
    "FedEM_SW": "mixture_SW",
    "AFL": "AFL",
    "FFL": "FFL",
    "IFCA": "IFCA",
    "APFL": "normal",
    "L2SGD": "normal",
    "FedAvg": "normal",
    "FedProx": "normal",
    "local": "normal",
    "pFedMe": "normal",
    "clustered": "normal",
    "stoCFL":"normal",
    "ICFL":"normal",
    "fedrc": "fedrc",               # tag
    "fedrc_Adam": "fedrc_Adam",
    "fedrc_DP": "fedrc_DP",
    "fedrc_tune": "fedrc_tune",
    "fedrc_TS": "fedrc_TS",
    "fedrc_SW": "fedrc_SW",
    "fedrc_LESW": "fedrc_LESW",
    "fedrc_LESWC": "fedrc_LESWC",
    "fedrc_ESW": "fedrc_ESW",
    "FeSEM": "FeSEM",        # tag
    "FeSEM_SW": "FeSEM",
    "FedSoft": "FedSoft",
    "FedGMM": "ACGmixture"
}

SHAKESPEARE_CONFIG = {
    "input_size": len(string.printable),
    "embed_size": 8,
    "hidden_size": 256,
    "output_size": len(string.printable),
    "n_layers": 2,
    "chunk_len": 80
}

CLASS_NUMBER = {
    'cifar10-c': 10,
    "cifar10-c-imbalance": 10,
    'cifar10-c-concept-only': 10,
    'cifar10-c-concept-5': 10,
    'cifar10-c-concept-label': 10,
    'cifar10-c-concept-feature': 10,
    'cifar10-c-concept-feature-label': 10,
    'cifar10-c-feature-only': 10,
    'cifar10-c-label-only': 10,
    'cifar10-c-2swap': 10,
    'cifar100-c-4swap': 100,
    'cifar100-c-2swap': 100,
    'cifar10-c-4swap': 10,
    'cifar10-c-noisy-02': 10,
    'cifar10-c-noisy-type2-02': 10,
    'cifar100-c-noisy-type2-04': 100,
    "cifar10-c-noisy-01": 10,
    'cifar10-c-noisy-type2-04': 10,
    'cifar10-c-noisy-04': 10,
    'cifar100-c-noisy-02': 100,
    'cifar100-c': 100,
    'fmnist-c': 10,
    "tiny-imagenet-c": 200,
    "tiny-imagenet-c-2swap": 200,
    "tiny-imagenet-c-4swap": 200,
    'shakespeare': 100,
    'airline': 2,
    'powersupply': 2,
    'elec': 2,
    # new
    'MNIST': 10,
    'CIFAR10': 10,
    'CIFAR100': 100,
    'FMNIST': 10
}

CHARACTERS_WEIGHTS = {
    '\n': 0.43795308843799086,
    ' ': 0.042500849608091536,
    ',': 0.6559597911540539,
    '.': 0.6987226398690805,
    'I': 0.9777491725556848,
    'a': 0.2226022051965085,
    'c': 0.813311655455682,
    'd': 0.4071860494572223,
    'e': 0.13455606165058104,
    'f': 0.7908671114133974,
    'g': 0.9532922255751889,
    'h': 0.2496906467588955,
    'i': 0.27444893060347214,
    'l': 0.37296488139109546,
    'm': 0.569937324017103,
    'n': 0.2520734570378263,
    'o': 0.1934141300462555,
    'r': 0.26035705948768273,
    's': 0.2534775933879391,
    't': 0.1876471355731429,
    'u': 0.47430062920373184,
    'w': 0.7470615815733715,
    'y': 0.6388302610200002
}

