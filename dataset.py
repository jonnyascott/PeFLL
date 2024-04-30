from collections import defaultdict
import itertools
import json
import numpy as np
import os
import random
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Subset, TensorDataset, DataLoader


def get_datasets(data_name, dataroot, normalize=True, val_size=10000, one_hot=True):
    """
    get_datasets returns train/val/test data splits of CIFAR10/100 datasets
    :param data_name: name of dataset, choose from [cifar10, cifar100]
    :param dataroot: root to data dir
    :param normalize: True/False to normalize the data
    :param val_size: validation split size (in #samples)
    :param one_hot: encode labels as one hot
    :return: train_set, val_set, test_set (tuple of pytorch dataset/subset)
    """

    if data_name == 'cifar10':
        normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        data_obj = CIFAR10
        num_classes = 10
    elif data_name == 'cifar100':
        normalization = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        data_obj = CIFAR100
        num_classes = 100
    else:
        raise ValueError("choose data_name from ['mnist', 'cifar10', 'cifar100']")

    trans = [transforms.ToTensor()]

    if normalize:
        trans.append(normalization)

    transform = transforms.Compose(trans)
    target_transform = None
    if one_hot:
        target_transform = transforms.Lambda(
            lambda y: torch.zeros(num_classes, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
        )

    dataset = data_obj(
        dataroot,
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform
    )

    test_set = data_obj(
        dataroot,
        train=False,
        download=True,
        transform=transform,
        target_transform=target_transform
    )

    train_size = len(dataset) - val_size

    data_class_idx = [np.where(np.array(dataset.targets) == i)[0] for i in range(num_classes)]
    for data_idx in data_class_idx:
        random.shuffle(data_idx)

    train_per_class = train_size // num_classes
    train_data_idx = list(itertools.chain.from_iterable([data_idx[:train_per_class] for data_idx in data_class_idx]))
    val_data_idx = list(itertools.chain.from_iterable([data_idx[train_per_class:] for data_idx in data_class_idx]))

    train_set = torch.utils.data.Subset(dataset, train_data_idx)
    val_set = torch.utils.data.Subset(dataset, val_data_idx)
    # train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    return train_set, val_set, test_set


def get_num_classes_samples(dataset):
    """
    extracts info about certain dataset
    :param dataset: pytorch dataset object
    :return: dataset info number of classes, number of samples, list of labels
    """
    # ---------------#
    # Extract labels #
    # ---------------#
    if isinstance(dataset, torch.utils.data.Subset):
        if isinstance(dataset.dataset.targets, list):
            data_labels_list = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            data_labels_list = dataset.dataset.targets[dataset.indices]
    else:
        if isinstance(dataset.targets, list):
            data_labels_list = np.array(dataset.targets)
        else:
            data_labels_list = dataset.targets
    classes, num_samples = np.unique(data_labels_list, return_counts=True)
    num_classes = len(classes)
    # print(num_classes, num_samples, data_labels_list)
    return num_classes, num_samples, data_labels_list


def gen_classes_per_node(dataset, num_users, classes_per_user=2, high_prob=0.5, low_prob=0.5):
    """
    creates the data distribution of each client
    :param dataset: pytorch dataset object
    :param num_users: number of clients
    :param classes_per_user: number of classes assigned to each client
    :param high_prob: highest prob sampled
    :param low_prob: lowest prob sampled
    :return: dictionary mapping between classes and proportions, each entry refers to other client
    """
    num_classes, num_samples, _ = get_num_classes_samples(dataset)

    # -------------------------------------------#
    # Divide classes + num samples for each user #
    # -------------------------------------------#
    # assert (classes_per_user * num_users) % num_classes == 0, "equal classes appearance is needed"
    count_per_class = (classes_per_user * num_users) // num_classes
    class_dict = {}
    for i in range(num_classes):
        # sampling alpha_i_c
        probs = np.random.uniform(low_prob, high_prob, size=count_per_class)
        # normalizing
        probs_norm = (probs / probs.sum()).tolist()
        class_dict[i] = {'count': count_per_class, 'prob': probs_norm}

    # -------------------------------------#
    # Assign each client with data indexes #
    # -------------------------------------#
    class_partitions = defaultdict(list)
    for i in range(num_users):
        c = []
        for _ in range(classes_per_user):
            class_counts = [class_dict[i]['count'] for i in range(num_classes)]
            max_class_counts = np.where(np.array(class_counts) == max(class_counts))[0]
            c.append(np.random.choice(max_class_counts))
            class_dict[c[-1]]['count'] -= 1
        assert len(set(c)) == classes_per_user
        class_partitions['class'].append(c)
        class_partitions['prob'].append([class_dict[i]['prob'].pop() for i in c])
    return class_partitions


def gen_classes_per_node_dirichlet(dataset, num_users, num_train_users, alpha_train, alpha_test=None,
                                   embedding_dir_path=None):
    """
        creates the data distribution of each client by sampling class proportions from dirichlet
        :param dataset: pytorch dataset object
        :param num_users: number of clients
        :param num_train_users: number of clients used for training
        :param alpha_train: dirichlet parameter for train clients
        :param alpha_test: dirichlet parameter for test clients
        :return: dictionary mapping between classes and proportions, each entry refers to other client
        """

    num_classes, num_samples, _ = get_num_classes_samples(dataset)

    train_user_class_proportions = np.random.dirichlet(alpha_train * np.ones(num_classes), size=num_train_users)
    if num_train_users < num_users:
        assert alpha_test is not None
        num_test_users = num_users - num_train_users
        test_user_class_proportions = np.random.dirichlet(alpha_test * np.ones(num_classes), size=num_test_users)
        user_class_proportions = np.vstack([train_user_class_proportions, test_user_class_proportions])
    else:
        user_class_proportions = train_user_class_proportions

    assert user_class_proportions.shape == (num_users, num_classes)
    normalized_user_class_proportions = user_class_proportions / user_class_proportions.sum(axis=0, keepdims=True)

    class_partitions = defaultdict(list)
    for i in range(num_users):
        class_partitions['class'].append(list(range(num_classes)))
        class_partitions['prob'].append(normalized_user_class_proportions[i])

    if embedding_dir_path is not None:
        np.save(f'{embedding_dir_path}/user_class_proportions.npy', user_class_proportions)

    return class_partitions


def gen_data_split(dataset, num_users, class_partitions):
    """
    divide data indexes for each client based on class_partition
    :param dataset: pytorch dataset object (train/val/test)
    :param num_users: number of clients
    :param class_partitions: proportion of classes per client
    :return: dictionary mapping client to its indexes
    """
    num_classes, num_samples, data_labels_list = get_num_classes_samples(dataset)

    # -------------------------- #
    # Create class index mapping #
    # -------------------------- #
    data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}

    # --------- #
    # Shuffling #
    # --------- #
    for data_idx in data_class_idx.values():
        random.shuffle(data_idx)

    # ------------------------------ #
    # Assigning samples to each user #
    # ------------------------------ #
    user_data_idx = [[] for i in range(num_users)]
    for usr_i in range(num_users):
        for c, p in zip(class_partitions['class'][usr_i], class_partitions['prob'][usr_i]):
            end_idx = int(num_samples[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]

    return user_data_idx


def gen_random_loaders(data_name, data_path, num_users, num_train_users, bz,
                       partition_type='by_class', classes_per_user=2, alpha_train=0.1, alpha_test=0.1,
                       embedding_dir_path=None):
    """
    generates train/val/test loaders of each client
    :param data_name: name of dataset, choose from [cifar10, cifar100]
    :param data_path: root path for data dir
    :param num_users: number of clients
    :param num_train_users: number of clients used to train
    :param bz: batch size
    :param partition_type: in ['by_class', 'dirichlet'], has class proportions are assigned to clients
    :param classes_per_user: number of classes assigned to each client
    :param alpha_train: dirichlet parameter for train clients
    :param alpha_test: dirichlet parameter for test clients
    :param embedding_dir_path: path
    :return: train/val/test loaders of each client, list of pytorch dataloaders
    """
    if data_name == 'femnist':
        return get_femnist(os.path.join(data_path, 'femnist'), bz)

    else:
        loader_params = {"batch_size": bz, "shuffle": False, "pin_memory": True, "num_workers": 0, "drop_last": False}
        dataloaders = []
        datasets = get_datasets(data_name, data_path, normalize=True)
        for i, d in enumerate(datasets):
            loader_params["drop_last"] = False
            # ensure same partition for train/test/val
            if i == 0:
                if partition_type == 'by_class':
                    cls_partitions = gen_classes_per_node(d, num_users, classes_per_user)
                elif partition_type == 'dirichlet':
                    cls_partitions = gen_classes_per_node_dirichlet(d, num_users, num_train_users, alpha_train,
                                                                    alpha_test, embedding_dir_path)
                else:
                    raise ValueError('partition_type must be in [by_class, dirichlet]')
                loader_params['shuffle'] = True
                loader_params["drop_last"] = True
            usr_subset_idx = gen_data_split(d, num_users, cls_partitions)
            # create subsets for each client
            subsets = list(map(lambda x: torch.utils.data.Subset(d, x), usr_subset_idx))
            # create dataloaders from subsets
            dataloaders.append(list(map(lambda x: torch.utils.data.DataLoader(x, **loader_params), subsets)))

        return dataloaders


def get_femnist(dataroot, bz):
    num_classes = 62
    val_split = 0.1
    img_size = 28

    train_dls, val_dls, test_dls = dict(), dict(), dict()
    for split in ['train', 'test']:
        for filename in os.listdir(os.path.join(dataroot, split)):
            with open(os.path.join(dataroot, split, filename), 'rb') as f:
                user_data_dic = json.load(f)

            node_ids = user_data_dic['users']
            for node_id in node_ids:
                xs = torch.tensor(user_data_dic['user_data'][node_id]['x'])
                xs = torch.reshape(xs, (-1, 1, img_size, img_size))
                xs = torch.nn.functional.pad(xs, (2, 2, 2, 2))
                ys = torch.tensor(user_data_dic['user_data'][node_id]['y'])
                ys = torch.nn.functional.one_hot(ys, num_classes).float()
                ds = TensorDataset(xs, ys)

                if split == 'train':
                    n = len(ds)
                    idxs = list(range(n))
                    np.random.shuffle(idxs)
                    val_idxs = sorted(idxs[:int(n * val_split)])
                    train_idxs = sorted(idxs[int(n * val_split):])
                    train_ds = Subset(ds, train_idxs)
                    val_ds = Subset(ds, val_idxs)

                    train_dls[node_id] = DataLoader(train_ds, batch_size=bz, shuffle=True, pin_memory=True,
                                                    num_workers=0, drop_last=False)
                    val_dls[node_id] = DataLoader(val_ds, batch_size=bz, shuffle=False, pin_memory=True,
                                                  num_workers=0)

                else:
                    test_ds = ds
                    test_dls[node_id] = DataLoader(test_ds, batch_size=bz, shuffle=False, pin_memory=True,
                                                   num_workers=0)

    keys = sorted(list(train_dls.keys()))
    idxs = list(range(len(keys)))
    np.random.shuffle(idxs)
    num_train = int(0.9 * len(keys))
    train_keys = [keys[i] for i in sorted(idxs[:num_train])]
    test_keys = [keys[i] for i in sorted(idxs[num_train:])]

    all_dls = [[], [], []]
    for key in train_keys + test_keys:
        all_dls[0].append(train_dls[key])
        all_dls[1].append(val_dls[key])
        all_dls[2].append(test_dls[key])

    return all_dls
