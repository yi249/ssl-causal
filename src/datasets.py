import os
import gzip
import random
import struct
from typing import Dict, List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as TF
from torchvision.transforms import v2
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from hps import Hparams
from utils import log_standardize, normalize

def _load_uint8(f):
    idx_dtype, ndim = struct.unpack("BBBB", f.read(4))[2:]
    shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data

def load_idx(path: str, dtype: str) -> np.ndarray:
    """Reads an array in IDX format from disk.
    Parameters
    ----------
    path : str
        Path of the input file. Will uncompress with `gzip` if path ends in '.gz'.
    Returns
    -------
    np.ndarray
        Output array of dtype ``uint8``.
    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    open_fcn = gzip.open if path.endswith(".gz") else open
    with open_fcn(path, "rb") as f:
        return _load_uint8(f)


def _get_paths(root_dir, train):
    prefix = "train" if train else "t10k"
    images_filename = prefix + "-images-idx3-ubyte.gz"
    labels_filename = prefix + "-labels-idx1-ubyte.gz"
    metrics_filename = prefix + "-morpho.csv"
    images_path = os.path.join(root_dir, images_filename)
    labels_path = os.path.join(root_dir, labels_filename)
    metrics_path = os.path.join(root_dir, metrics_filename)
    return images_path, labels_path, metrics_path


def load_morphomnist_like(
    root_dir, train: bool = True, columns=None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Args:
        root_dir: path to data directory
        train: whether to load the training subset (``True``, ``'train-*'`` files) or the test
            subset (``False``, ``'t10k-*'`` files)
        columns: list of morphometrics to load; by default (``None``) loads the image index and
            all available metrics: area, length, thickness, slant, width, and height
    Returns:
        images, labels, metrics
    """
    images_path, labels_path, metrics_path = _get_paths(root_dir, train)
    images = load_idx(images_path, "uint8")
    #labels = load_idx(labels_path, "float64")

    if columns is not None and "index" not in columns:
        usecols = ["index"] + list(columns)
    else:
        usecols = columns
    labels = pd.read_csv(metrics_path, usecols=usecols, index_col="index")
    return images, labels

def one_hot_with_nans(input_tensor, num_classes):
    input_tensor += 1
    long = torch.from_numpy(np.array(input_tensor).astype(np.uint8)).long()
    one_hot = F.one_hot(long, num_classes=num_classes+1)[:,1:].to(torch.float64)
    zero_rows_mask = torch.all(one_hot == 0, dim=1)
    one_hot[zero_rows_mask] = float('nan')
    return one_hot

class MorphoMNIST(Dataset):
    def __init__(
        self,
        root_dir: str,
        train: bool = True,
        transform: Optional[torchvision.transforms.Compose] = None,
        columns: Optional[List[str]] = None,
        norm: Optional[str] = None,
        concat_pa: bool = True,
        beginning: bool = False,
    ):
        self.train = train
        self.transform = transform
        self.columns = columns
        self.concat_pa = concat_pa
        self.norm = norm

        # cols_not_digit = [c for c in self.columns if c != "digit"]
        images, labels = load_morphomnist_like(
            root_dir, train, self.columns
        )

        self.images = torch.from_numpy(np.array(images)).unsqueeze(1)

        if self.columns is None:
            self.columns = labels.columns
        self.samples = {k: torch.tensor(labels[k]) for k in self.columns}

        if beginning == True:
            labelled = 0
            length = len(self.samples[list(self.samples.keys())[0]])
            for i in range(length):
                if all(v[i] == v[i] for v in self.samples.values()):
                    labelled += 1

            # labelled = len(self.samples[list(self.samples.keys())[0]])
            print('labelled', labelled)
            self.images = self.images[:labelled]

        self.min_max = {
            "thickness": [0.87598526, 6.255515],
            "intensity": [66.601204, 254.90317],
        }

        for k, v in self.samples.items():  # optional preprocessing
            if k in ["thickness", "intensity"]:
                print(f"{k} normalization: {norm}")
                if norm == "[-1,1]":
                    self.samples[k] = normalize(
                        v, x_min=self.min_max[k][0], x_max=self.min_max[k][1]
                    )
                elif norm == "[0,1]":
                    self.samples[k] = normalize(
                        v, x_min=self.min_max[k][0], x_max=self.min_max[k][1], zero_one=True
                    )
                elif norm == None:
                    pass
                else:
                    NotImplementedError(f"{norm} not implemented.")
            elif k == "digit":
                print("digit processed")
                self.samples[k] = one_hot_with_nans(v, num_classes=10).squeeze()

        print(f"#samples: {len(labels)}\n")

        # self.samples.update({"digit": self.labels})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = {}
        sample["x"] = self.images[idx].squeeze()

        if self.transform is not None:
            sample["x"] = self.transform(sample["x"])

        if self.concat_pa:
            sample["pa"] = torch.cat(
                [
                    v[idx] if k == "digit" else torch.tensor([v[idx]])
                    for k, v in self.samples.items()
                ],
                dim=0,
            )
        else:
            try:
                sample["fgcol"] = torch.cat([torch.tensor([v[idx]])
                                for k,v in self.samples.items() if "fg" in k], dim=0)
            except:
                sample["fgcol"] = torch.tensor([float('nan')]).repeat(3)
            try:
                sample["bgcol"] = torch.cat([torch.tensor([v[idx]])
                                for k, v in self.samples.items() if "bg" in k], dim=0)
            except:
                sample["bgcol"] = torch.tensor([float('nan')]).repeat(3)

            for k,v in self.samples.items():
                if "fg" not in k and "bg" not in k:
                    try:
                        sample.update({k: v[idx].to(torch.float64) if k == "digit"
                                       else torch.tensor([v[idx]])})

                    except:
                        if "digit" in k:
                            sample.update({k: torch.tensor([float('nan')]).repeat(10)})
                        else:
                            sample.update({k: torch.tensor([float('nan')])})
        return sample


def morphomnist(args: Hparams, beginning) -> Dict[str, MorphoMNIST]:
    # Load datapad
    if not args.data_dir:
        args.data_dir = "../morphomnist/"

    augmentation = {
        "train": TF.Compose(
            [
                TF.RandomCrop((args.input_res, args.input_res), padding=args.pad),
            ]
        ),
        "eval": TF.Compose(
            [
                TF.Pad(padding=2),  # (32, 32)
            ]
        ),
    }

    datasets = {}
    for split in ["train", "valid", "test"]:
        datasets[split] = MorphoMNIST(
            root_dir=args.data_dir,
            train=(split == "train"),  # test set is valid set
            columns=args.parents_x,
            norm=args.context_norm,
            concat_pa=args.concat_pa,
            beginning=beginning
        )
    return datasets


class MIMICMetadata(TypedDict):
    age: float  # age in years
    sex: int  # 0 -> male , 1 -> female
    race: int  # 0 -> white , 1 -> asian , 2 -> black


def read_mimic_from_df(
    idx: int, df: pd.DataFrame, data_dir: str
) -> Tuple[Image.Image, torch.Tensor, MIMICMetadata]:
    """Get a single data point from the MIMIC-CXR dataframe.

    References:
    Written by Charles Jones.
    https://github.com/biomedia-mira/chexploration/blob/main/notebooks/mimic.sample.ipynb

    Args:
        idx (int): Index of the data point to retrieve.
        df (pd.DataFrame): Dataframe containing the MIMIC-CXR data.
        data_dir (str): Path to the directory containing the preprocessed data.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the image and binary label.
            label 0 represents no finding, 1 represents pleural effusion.
    """
    img_path = os.path.join(data_dir, df.iloc[idx]["path_preproc"])
    img = Image.open(img_path)  # .convert("RGB")
    # if df.iloc[idx]["disease"] == "Pleural Effusion":
    #     label = torch.tensor(1)
    # elif df.iloc[idx]["disease"] == "No Finding":
    #     label = torch.tensor(0)
    # else:
    #     raise ValueError(
    #         f"Invalid label {df.iloc[idx]['disease']}.",
    #         "We expect either 'pleural_effusion' or 'no_finding'.",
    #     )

    label = df.iloc[idx]["disease_label"]

    age = df.iloc[idx]["age"]
    sex = df.iloc[idx]["sex_label"]
    race = df.iloc[idx]["race_label"]

    meta = MIMICMetadata(age=age, sex=sex, race=race)
    return img, label, meta


class MIMIC(Dataset):
    def __init__(
        self,
        split_path,
        data_dir,
        cache=False,
        transform=None,
        parents_x=None,
        concat_pa=False,
    ):
        super().__init__()
        self.concat_pa = concat_pa
        self.parents_x = parents_x
        self.split_df = pd.read_csv(split_path)
        # # remove rows whose disease label is neither No Finding nor Pleural Effusion
        # self.split_df = split_df[
        #     (split_df["disease"] == "No Finding")
        #     | (split_df["disease"] == "Pleural Effusion")
        # ].reset_index(drop=True)

        self.data_dir = data_dir
        self.cache = cache
        self.transform = transform

        if self.cache:
            self.imgs = []
            self.labels = []
            self.meta = []
            for idx, _ in tqdm(
                self.split_df.iterrows(), total=len(self.split_df), desc="Caching MIMIC"
            ):
                assert isinstance(idx, int)
                img, label, meta = read_mimic_from_df(idx, self.split_df, self.data_dir)
                self.imgs.append(img)
                self.labels.append(label)
                self.meta.append(meta)

    def __len__(self):
        return len(self.split_df)
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        if self.cache:
            img = self.imgs[idx]
            label = self.labels[idx]
            meta = self.meta[idx]
        else:
            img, label, meta = read_mimic_from_df(idx, self.split_df, self.data_dir)
        sample = {}
        sample["x"] = self.transform(img)

        try:
            sample["finding"] = label
        except:
            sample["finding"] = torch.tensor([float('nan')])

        try:
            sample["age"] = meta['age']
        except:
            sample["age"] = torch.tensor([float('nan')])

        try:
            sample["sex"] = meta['sex']
        except:
            sample["sex"] = torch.tensor([float('nan')])

        try:
            sample["race"] = meta['race']
        except:
            sample["race"] = torch.tensor([float('nan')]).repeat(3)
        sample = preprocess_mimic(sample)
        if self.concat_pa:
            sample["pa"] = torch.cat(
                [sample[k] for k in self.parents_x],
                dim=0,
            )
        return sample



def preprocess_mimic(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
    for k, v in sample.items():
        if k != "x":
            sample[k] = torch.tensor([v])
            if k == "race":
                try:
                    sample[k] = F.one_hot(sample[k], num_classes=3).squeeze()
                except:
                    sample[k] = torch.tensor([float('nan')]).repeat(3)
            elif k == "age":
                try:
                    sample[k] = sample[k] / 100 * 2 - 1  # [-1,1]
                except:
                    sample[k] = torch.tensor([float('nan')])
    return sample


def mimic(
    args: Hparams,
    augmentation: Optional[Dict[str, torchvision.transforms.Compose]] = None,
) -> Dict[str, MIMIC]:
    if augmentation is None:
        augmentation = {}
        augmentation["train_lab"] = v2.Compose(
            [
            TF.Resize((args.input_res, args.input_res), antialias=None),
            TF.PILToTensor(),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 5.))], p=0.25),
            v2.ColorJitter(brightness=0.3, contrast=0.3),
            v2.RandomRotation(90),
            v2.RandomHorizontalFlip(p=0.25),
        ]
        )
        augmentation["other"] = TF.Compose(
            [
            TF.Resize((args.input_res, args.input_res), antialias=None),
            TF.PILToTensor(),
            ]
        )

    datasets = {}
    if args.random:
        for split in ["train", "valid", "test"]:
            datasets[split] = MIMIC(
                data_dir=os.path.join(args.data_dir, "data"),
                split_path=os.path.join(args.data_dir, f"meta/random/{split}_{args.labelled}.csv"),
                cache=False,
                parents_x=args.parents_x,  # ["age", "race", "sex", "finding"],
                concat_pa=(True if not hasattr(args, "concat_pa") else args.concat_pa),
                transform=augmentation["other"],
            )
    else:
        for split in ["train_lab", "train_unlab", "valid", "test"]:
            datasets[split] = MIMIC(
                data_dir=os.path.join(args.data_dir, "data"),
                split_path=os.path.join(args.data_dir, f"meta/missing/{split}_{args.labelled}.csv"),
                cache=False,
                parents_x=args.parents_x,  # ["age", "race", "sex", "finding"],
                concat_pa=(True if not hasattr(args, "concat_pa") else args.concat_pa),
                transform=augmentation["other"]#augmentation["train_lab"] if split == "train_lab" else augmentation["other"],
            )
    return datasets
