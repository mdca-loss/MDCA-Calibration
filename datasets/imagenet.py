# taken from https://github.com/torrvision/focal_calibration/blob/main/Data/tiny_imagenet.py
"""
Create train, val, test iterators for Tiny ImageNet.
Train set size: 100000
Val set size: 10000
Test set size: 10000
Number of classes: 200
Link: https://tiny-imagenet.herokuapp.com/
"""

import os

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import glob
from torch.utils.data import Dataset
from PIL import Image
from torch.utils import data

import random

EXTENSION = 'JPEG'
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = 'wnids.txt'
VAL_ANNOTATION_FILE = 'val_annotations.txt'


class TinyImageNet(Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """
    def __init__(self, root='data/tiny-imagenet-200', split='train', transform=None, target_transform=None, in_memory=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.in_memory = in_memory
        self.split_dir = os.path.join(root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing

        # build class label - number mapping
        with open(os.path.join(self.root, CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

        if self.split == 'test':
            return img
        else:
            # file_name = file_path.split('/')[-1]
            return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def read_image(self, path):
        img = Image.open(path)
        if (img.mode == 'L'):
            img = img.convert('RGB')
        return self.transform(img) if self.transform else img

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
])

transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
])

def get_train_valid_test_loader(args):
    train_set = TinyImageNet(   split='train',
                               transform=transform_train,
                               in_memory=False)
    print(train_set)

    val_set = TinyImageNet( split='train',
                               transform=transform_test,
                               in_memory=False)

    # create a val set from training set
    idxs = list(range(len(train_set)))
    random.seed(args.seed)
    random.shuffle(idxs)
    split = int(0.1 * len(idxs))
    train_idxs, valid_idxs = idxs[split:], idxs[:split]

    train_sampler = data.SubsetRandomSampler(train_idxs)
    val_sampler = data.SequentialSampler(valid_idxs)

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=args.workers, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.test_batch_size, num_workers=args.workers, drop_last=False, sampler=val_sampler)

    test_set = TinyImageNet(    split='val',
                               transform=transform_test,
                               in_memory=False)
    print(test_set)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    return train_loader, val_loader, test_loader

def get_datasets(args):
    train_set = TinyImageNet(   split='train',
                               transform=None,
                               in_memory=False)
    test_set = TinyImageNet(    split='val',
                               transform=None,
                               in_memory=False)
    return train_set, test_set

def get_transforms():
    return transform_train, transform_test