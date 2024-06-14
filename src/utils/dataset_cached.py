import errno
import os
import PIL
from functools import reduce
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
import torchvision.transforms as transforms


def split_celeba(X, y, sup_frac, validation_num):
    """
    splits celeba
    """

    # validation set is the last 10,000 examples
    X_valid = X[-validation_num:]
    y_valid = y[-validation_num:]

    X = X[0:-validation_num]
    y = y[0:-validation_num]

    if sup_frac == 0.0:
        return None, None, X, y, X_valid, y_valid

    if sup_frac == 1.0:
        return X, y, None, None, X_valid, y_valid

    split = int(sup_frac * len(X))
    X_sup = X[0:split]
    y_sup = y[0:split]
    X_unsup = X[split:]
    y_unsup = y[split:]

    return X_sup, y_sup, X_unsup, y_unsup, X_valid, y_valid


CELEBA_LABELS = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
                 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', \
                 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', \
                 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
                 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', \
                 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

CELEBA_EASY_LABELS = ['Arched_Eyebrows', 'Bags_Under_Eyes', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
                      'Bushy_Eyebrows', 'Chubby', 'Eyeglasses', 'Heavy_Makeup', 'Male', \
                      'No_Beard', 'Pale_Skin', 'Receding_Hairline', 'Smiling', 'Wavy_Hair', 'Wearing_Necktie', 'Young']


class CELEBACached(CelebA):
    """
    a wrapper around CelebA to load and cache the transformed data
    once at the beginning of the inference
    """
    # static class variables for caching training data
    train_data_sup, train_labels_sup = None, None
    train_data_unsup, train_labels_unsup = None, None
    train_data, test_labels = None, None
    prior = torch.ones(1, len(CELEBA_EASY_LABELS)) / 2
    fixed_imgs = None
    validation_size = 20000
    data_valid, labels_valid = None, None

    def prior_fn(self):
        return CELEBACached.prior

    def clear_cache():
        CELEBACached.train_data, CELEBACached.test_labels = None, None

    def __init__(self, mode, sup_frac=None, *args, **kwargs):
        super(CELEBACached, self).__init__(split='train' if mode in ["sup", "unsup", "valid"] else 'test', *args,
                                           **kwargs)
        self.sub_label_inds = [i for i in range(len(CELEBA_LABELS)) if CELEBA_LABELS[i] in CELEBA_EASY_LABELS]
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        assert mode in ["sup", "unsup", "test", "valid"], "invalid train/test option values"

        if mode in ["sup", "unsup", "valid"]:

            if CELEBACached.train_data is None:
                print("Splitting Dataset")

                CELEBACached.train_data = self.filename
                CELEBACached.train_targets = self.attr

                CELEBACached.train_data_sup, CELEBACached.train_labels_sup, \
                    CELEBACached.train_data_unsup, CELEBACached.train_labels_unsup, \
                    CELEBACached.data_valid, CELEBACached.labels_valid = \
                    split_celeba(CELEBACached.train_data, CELEBACached.train_targets,
                                 sup_frac, CELEBACached.validation_size)

            if mode == "sup":
                self.data, self.targets = CELEBACached.train_data_sup, CELEBACached.train_labels_sup
                CELEBACached.prior = torch.mean(self.targets[:, self.sub_label_inds].float(), dim=0)
            elif mode == "unsup":
                self.data = CELEBACached.train_data_unsup
                # making sure that the unsupervised labels are not available to inference
                self.targets = CELEBACached.train_labels_unsup * np.nan
            else:
                self.data, self.targets = CELEBACached.data_valid, CELEBACached.labels_valid

        else:
            self.data = self.filename
            self.targets = self.attr

        # create a batch of fixed images
        if CELEBACached.fixed_imgs is None:
            temp = []
            for i, f in enumerate(self.data[:64]):
                temp.append(
                    self.transform(PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", f))))
            CELEBACached.fixed_imgs = torch.stack(temp, dim=0)

    def __getitem__(self, index):

        X = self.transform(
            PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.data[index])))

        target = self.targets[index].float()
        target = target[self.sub_label_inds]

        return X, target

    def __len__(self):
        return len(self.data)


def setup_data_loaders(use_cuda, batch_size, sup_frac=1.0, root=None, cache_data=False, **kwargs):
    """
        helper function for setting up pytorch data loaders for a semi-supervised dataset
    :param use_cuda: use GPU(s) for training
    :param batch_size: size of a batch of data to output when iterating over the data loaders
    :param sup_frac: fraction of supervised data examples
    :param cache_data: saves dataset to memory, prevents reading from file every time
    :param kwargs: other params for the pytorch data loader
    :return: three data loaders: (supervised data for training, un-supervised data for training,
                                  supervised data for testing)
    """

    if root is None:
        root = get_data_directory(__file__)
    if 'num_workers' not in kwargs:
        kwargs = {'num_workers': 4, 'pin_memory': True}
    cached_data = {}
    loaders = {}

    # clear previous cache
    CELEBACached.clear_cache()

    if sup_frac == 0.0:
        modes = ["unsup", "test"]
    elif sup_frac == 1.0:
        modes = ["sup", "test", "valid"]
    else:
        modes = ["unsup", "test", "sup", "valid"]

    for mode in modes:
        cached_data[mode] = CELEBACached(root=root, mode=mode, download=True, sup_frac=sup_frac)
        loaders[mode] = DataLoader(cached_data[mode], batch_size=batch_size, shuffle=True, **kwargs)
    return loaders
