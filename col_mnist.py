import os
import sys
sys.path.insert(0,  os.path.abspath("src"))
import torchvision.transforms as TF
import torch
import numpy as np
import pandas as pd
from torch.distributions import Beta, Uniform
import tqdm
from argparse import ArgumentParser
import numpy as np
from torch.utils.data import DataLoader
import random
import src.datasets as setup
import idx2numpy
import gzip

parser = ArgumentParser()

parser.add_argument("--size", help="proportion of data to be labelled", type=int, default=60000)
parser.add_argument("--random", help="whether to remove labels randomly", action="store_true", default=False)

args = parser.parse_args()

data_dir = "datasets/morphomnist"
main_dir = 'datasets/col_morphomnist/'
if not os.path.exists(main_dir):
    os.mkdir(main_dir)

if args.random:
    if not os.path.exists(main_dir+"random"):
        os.mkdir(main_dir+"random")
    dir_name = main_dir+"random/"+str(args.size)

else:
    if not os.path.exists(main_dir + "missing"):
        os.mkdir(main_dir + "missing")
    dir_name = main_dir+"missing/"+str(args.size)

if not os.path.exists(dir_name):
    os.mkdir(dir_name)

augmentation = {
        "train": TF.Compose(
            [
                TF.RandomCrop((32, 32), padding=4),
            ]
        ),
        "eval": TF.Compose(
            [
                TF.Pad(padding=2),  # (32, 32)
            ]
        ),
    }

datasets = {}
for split in ["train", "test"]:
    datasets[split] = setup.MorphoMNIST(
        root_dir=data_dir,
        train=(split == "train"),
        transform=augmentation[("eval" if split != "train" else split)],
        columns=["thickness", "intensity", "digit"],
        concat_pa=True
    )

bs=1
dataloaders = {
        "train": DataLoader(datasets["train"], shuffle=True, batch_size=bs),
        "test": DataLoader(datasets["test"], shuffle=False),
    }

#### COLOUR GENERATION ####

# generator used in paper
# beta(4,2) or beta(2,4) for each channel depending on digit (mod 3)
def colour_generator(digit):
    option = digit % 3
    a = Beta(torch.FloatTensor([4]), torch.FloatTensor([2]))
    b = Beta(torch.FloatTensor([2]), torch.FloatTensor([4]))
    if option == 0:
        r = a.sample()
        g = b.sample()
        b = b.sample()
    elif option == 1:
        r = b.sample()
        g = a.sample()
        b = b.sample()
    else:
        r = b.sample()
        g = b.sample()
        b = a.sample()
    return torch.cat((r, g, b)) * 255.

# same as above but with fgcol depending on both digit and bgcol
def colour_generator_fg(digit, bgcol):
    option = digit % 3
    a = Beta(torch.FloatTensor([4]), torch.FloatTensor([2]))
    b = Beta(torch.FloatTensor([2]), torch.FloatTensor([4]))
    if option == 0:
        r = torch.tensor(np.mean([a.sample(), bgcol[0]]))
        g = torch.tensor(np.mean([b.sample(), bgcol[1]]))
        b = torch.tensor(np.mean([b.sample(), bgcol[2]]))
    elif option == 1:
        r = torch.tensor(np.mean([b.sample(), bgcol[0]]))
        g = torch.tensor(np.mean([a.sample(), bgcol[1]]))
        b = torch.tensor(np.mean([b.sample(), bgcol[2]]))
    else:
        r = torch.tensor(np.mean([b.sample(), bgcol[0]]))
        g = torch.tensor(np.mean([b.sample(), bgcol[1]]))
        b = torch.tensor(np.mean([a.sample(), bgcol[2]]))

    return torch.cat((r.reshape(1), g.reshape(1), b.reshape(1))) * 255.

# uniform distribution over each channel
def colour_generator_uniform():
    dist = Uniform(torch.tensor([0.0]), torch.tensor([255.0]))
    r = dist.sample()
    g = dist.sample()
    b = dist.sample()
    return torch.cat((r, g, b))

def none_fn(values, use_all, probability=0):
    shape = values.shape
    if np.random.rand() < probability or use_all==True:
        return values
    else:
        return np.full(shape, None)

def gen_fgbgcol_data(loader, noise, train):
    tot_iters = len(loader)
    for i, batch in enumerate(tqdm.tqdm(loader, total=tot_iters)):
        if train and args.random:
            prob = args.size/len(datasets["train"])
        x, targets = batch.values()
        digits = torch.argmax(targets[:, 2:], dim=1)
        assert len(
            x.size()) == 4, 'Something is wrong, size of input x should be 4 dimensional (B x C x H x W; perhaps number of channels is degenrate? If so, it should be 1)'
        digits = digits.cpu().numpy()
        bs = targets.shape[0]

        # create mask of digit
        mask = (((x * 255) > 0) * 255).type('torch.FloatTensor')
        x_rgb = x.expand(-1, 3, -1, -1)
        x_rgb_fg = 1. * x_rgb


        # generate background colour
        bg = (255 - mask.expand(-1, 3, -1, -1))
        c_bg = torch.ones(bs, 3) / 255. * torch.stack([colour_generator(digit + 1) for digit in digits])

        if train and args.random:
            targets = np.concatenate((none_fn(c_bg.numpy() * 2 - 1, False, probability=prob), targets), 1)
        else:
            targets = np.concatenate((c_bg.numpy() * 2 - 1, targets), 1)
        bg = bg.permute(1, 2, 3, 0)
        bg[0] = bg[0] * c_bg[:, 0]
        bg[1] = bg[1] * c_bg[:, 1]
        bg[2] = bg[2] * c_bg[:, 2]
        bg = bg.permute(3, 0, 1, 2)


        # generate foreground (digit) colour
        c_fg = torch.ones(bs, 3) / 255. * torch.stack([colour_generator(digit) for digit in digits])
        # c_fg = torch.ones(bs,3)/255. * torch.stack([colour_generator_fg(digit, bgcol) for digit, bgcol in zip(digits,c_bg)])
        if train and args.random:
            targets = np.concatenate((none_fn(c_fg.numpy() * 2 - 1, False, probability=prob), targets), 1)
        else:
            targets = np.concatenate((c_fg.numpy() * 2 - 1, targets), 1)
        x_rgb_fg = x_rgb_fg.permute(1, 2, 3, 0)
        x_rgb_fg[0] = x_rgb_fg[0] * c_fg[:, 0]
        x_rgb_fg[1] = x_rgb_fg[1] * c_fg[:, 1]
        x_rgb_fg[2] = x_rgb_fg[2] * c_fg[:, 2]
        x_rgb_fg = x_rgb_fg.permute(3, 0, 1, 2)

        targets = targets[:, :8]
        targets = np.concatenate((targets, np.array([digits])), 1)

        x_rgb = x_rgb_fg + bg
        x_rgb = x_rgb + torch.tensor((noise) * np.random.randn(*x_rgb.size())).type('torch.FloatTensor')
        x_rgb = torch.clamp(x_rgb, 0., 255.)
        if i == 0:
            col_data_x = np.zeros((bs * tot_iters, *x_rgb.size()[1:])).astype(np.uint8)
            col_data_y = np.zeros((bs * tot_iters, targets.shape[0], targets.shape[1]))
        col_data_x[i * bs: (i + 1) * bs] = x_rgb
        col_data_y[i * bs: (i + 1) * bs] = targets[:, :9]

    return col_data_x, np.squeeze(col_data_y)

train_data_x, train_data_y = gen_fgbgcol_data(dataloaders['train'], noise=10., train=True)
test_data_x, test_data_y = gen_fgbgcol_data(dataloaders['test'], noise=10., train=False)

if not args.random:
    train_data_y = train_data_y[:args.size,:]

def gzip_file(filename):
    with open(filename, 'rb') as f_in:
        with gzip.open(filename+'.gz', 'wb', compresslevel=1) as f_out:
            f_out.writelines(f_in)

columns = ['fg_r', 'fg_g', 'fg_b', 'bg_r', 'bg_g', 'bg_b', 'thickness', 'intensity', 'digit']

idx2numpy.convert_to_file(dir_name + '/train-images-idx3-ubyte', train_data_x)
pd.DataFrame(train_data_y, columns=columns).to_csv(dir_name + '/train-morpho.csv', index=True, index_label='index')

idx2numpy.convert_to_file(dir_name + '/t10k-images-idx3-ubyte', test_data_x)
pd.DataFrame(test_data_y, columns=columns).to_csv(dir_name + '/t10k-morpho.csv', index=True, index_label='index')

gzip_file(dir_name + '/train-images-idx3-ubyte')
gzip_file(dir_name + '/t10k-images-idx3-ubyte')

print("Your files are saved in", dir_name)