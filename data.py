import os
import random
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.datasets import CIFAR100
from torch.utils.data import Dataset

def tensor_to_image(tensor):
    """
    Convert a torch tensor into a numpy ndarray for visualization.

    :param tensor: a torch tensor of shape (3, H, W) with elements in the range [0, 1]

    :return: a uint8 numpy array of shape (H, W, 3)
    """
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    return ndarr


def _extract_tensors(dataset, num=None, x_dtype=torch.float64):
    """
    Convert data to tensor and make it range(0, 1)

    :param dataset: torchvision.datasets.CIFAR100
    :param num: number of images to extract
    :param x_dtype: dtype for input x

    :return: tensor of x, y
    """

    # x: (num_samples, 32, 32, 3) -> (num_samples, 3, 32, 32) -> divide by 255: range(0, 1)
    x = torch.tensor(dataset.data, dtype=x_dtype).permute(0, 3, 1, 2).div_(255)
    y = torch.tensor(dataset.targets, dtype=torch.int64)

    if num is not None:
        if num <= 0 or num > x.shape[0]:
            raise ValueError(f"Invalid value, num must be in range [0, {x.shape}]")
        x = x[:num]
        y = y[:num]

    return x, y


def cifar100(num_train=None, num_test=None, from_server=True, x_dtype=torch.float32):
    """
    :param num_train: number of training data
    :param num_test: number of test data
    :param from_server: get CIFAR-100 dataset from server if True
    :param x_dtype: dtype for input x

    :return: x_train, y_train, x_test, y_test
    """

    download = not os.path.isdir('data')
    # x_train: (50000, 32, 32, 3)
    # y_train: (50000,)
    dataset_train = CIFAR100(root='./data', train=True, download=download)
    # x_test: (10000, 32, 32, 3)
    # y_test: (10000,)
    dataset_test = CIFAR100(root='./data', train=False, download=download)

    x_train, y_train = _extract_tensors(dataset_train, num_train, x_dtype=x_dtype)
    x_test, y_test = _extract_tensors(dataset_test, num_test, x_dtype=x_dtype)

    return x_train, y_train, x_test, y_test



def norm_split_cifar100(validation_ratio=0.2, dtype=torch.float32):

    x_train, y_train, x_test, y_test = cifar100(x_dtype=dtype)

    # Subtract mean for every (R, G, B) channels
    mean_img = x_train.mean(dim=(0,2,3), keepdim=True)
    x_train -= mean_img
    x_test -= mean_img

    # Take the validation set from the training set
    random_idxs = torch.randperm(x_train.shape[0])
    num_training = int(x_train.shape[0] * (1 - validation_ratio))
    num_validation = x_train.shape[0] - num_training

    data_dict = {}
    data_dict["x_train"] = x_train[random_idxs[:num_training]]
    data_dict["y_train"] = y_train[random_idxs[:num_training]]
    data_dict["x_val"] = x_train[random_idxs[num_training:num_training + num_validation]]
    data_dict["y_val"] = y_train[random_idxs[num_training:num_training + num_validation]]
    data_dict["x_test"] = x_test
    data_dict["y_test"] = y_test

    return data_dict


def visualize_cifar100(dtype=torch.float32):

    x_train, y_train, _, _ = cifar100(x_dtype=dtype)

    # Visualize some examples from CIFAR100
    cifar100_classes = [
        "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
        "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
        "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
        "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
        "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck",
        "pine_tree",
        "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
        "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
        "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger",
        "tractor",
        "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
    ]

    samples_per_class = 12
    samples = []

    # show images for the first 10 classes
    for y, cls in enumerate(cifar100_classes[:10]):
        plt.text(-4, 34 * y + 18, cls, ha="right")
        (idxs,) = (y_train == y).nonzero(as_tuple=True)
        for i in range(samples_per_class):
            idx = idxs[random.randrange(idxs.shape[0])].item()
            samples.append(x_train[idx])
    img = torchvision.utils.make_grid(samples, nrow=samples_per_class)
    plt.imshow(tensor_to_image(img))
    plt.axis("off")
    plt.show()


class CIFAR100Dataset(Dataset):
    def __init__(self, x, y, transforms:torchvision.transforms=None):
        self.x = x
        self.y = y
        self.transforms = transforms

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):

        if self.transforms is not None:
            self.x[idx] = self.transforms(self.x[idx])

        return self.x[idx], self.y[idx]