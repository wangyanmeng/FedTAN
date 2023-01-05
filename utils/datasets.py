import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from torch.utils.data import Dataset


# iid / non-iid data distribution
def CIFAR10_iid(dataset, num_users, weight_set):
    """
    Sample I.I.D. client data from CIFAR-10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        num_items = int(np.round(len(dataset)*weight_set[i]))
        dict_users[i] = set(np.random.choice(all_idxs, num_items,replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def CIFAR10_noniid(dataset, num_users, weight_set): # be controlled by weight
    """
    Sample non-I.I.D client data from CIFAR-10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {i: np.array([]) for i in range(num_users)}
    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # sort labels
    idxs_labels = np.vstack((np.arange(len(dataset)), labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign shards/client
    start_index = 0
    for i in range(num_users):
        rand_set = set( np.arange(start_index, start_index + int(np.round(len(dataset) * weight_set[i])))  )
        dict_users[i] = np.concatenate((dict_users[i], idxs[list(rand_set)]), axis=0)
        start_index = start_index + int(np.round(len(dataset)*weight_set[i]))

    return dict_users

#datasets
def dataset_preprocess(node_number, weight_set, data_distribution, batch_size, workers):
    # dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True)

    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))

    train_all_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= batch_size, shuffle=False,
        num_workers= workers, pin_memory=True)

    test_all_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False,
        num_workers= workers, pin_memory=True)

    # iid / non-iid data distribution
    if data_distribution == 1:  # iid
        node_groups = CIFAR10_iid(train_dataset, node_number, weight_set)
    else:  # non-iid
        node_groups = CIFAR10_noniid(train_dataset, node_number, weight_set)

    return train_dataset, train_all_loader, test_all_loader, node_groups


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)