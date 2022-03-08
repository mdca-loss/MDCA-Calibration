import random

from torchvision import datasets
from torchvision import transforms
from torch.utils import data

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# train set = 45000
# val set = 5000
# test set = 10000
def get_train_valid_test_loader(args):
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)

    # create a val set from training set
    idxs = list(range(len(train_set)))
    random.seed(args.seed)
    random.shuffle(idxs)
    split = int(0.1 * len(idxs))
    train_idxs, valid_idxs = idxs[split:], idxs[:split]

    train_sampler = data.SubsetRandomSampler(train_idxs)
    val_sampler = data.SubsetRandomSampler(valid_idxs)

    train_loader = data.DataLoader(train_set, batch_size=args.train_batch_size, num_workers=args.workers, sampler=train_sampler)
    val_loader = data.DataLoader(val_set, batch_size=args.test_batch_size, num_workers=args.workers, sampler=val_sampler, drop_last=False)

    test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    test_loader = data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    return train_loader, val_loader, test_loader

def get_datasets(args):
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=None)
    return trainset, testset

def get_transforms():
    return transform_train, transform_test