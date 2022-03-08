import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from torchvision import datasets
from torchvision import transforms
from torch.utils import data

# means and standard deviations ImageNet because the network is pretrained
means, stds = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# Define transforms to apply to each image
transf = transforms.Compose([ #transforms.Resize(227),      # Resizes short size of the PIL image to 256
                              transforms.CenterCrop(224),  # Crops a central square patch of the image 224 because torchvision's AlexNet needs a 224x224 input!
                              transforms.ToTensor(), # Turn PIL Image to torch.Tensor
                              transforms.Normalize(means,stds) # Normalizes tensor with mean and standard deviation
])

path_dict = {
    "photo" : 'data/Homework3-PACS/PACS/photo',
    "art" : 'data/Homework3-PACS/PACS/art_painting',
    "cartoon" : 'data/Homework3-PACS/PACS/cartoon',
    "sketch" : 'data/Homework3-PACS/PACS/sketch'
}

def get_train_valid_test_loader(args, target_type="art"):
    # Prepare Pytorch train/test Datasets
    train_set = datasets.ImageFolder(path_dict["photo"], transform=transf)
    test_set = datasets.ImageFolder(path_dict[target_type], transform=transf)

    train_loader = data.DataLoader(train_set, batch_size=args.train_batch_size, num_workers=args.workers, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    return train_loader, test_loader, test_loader

def get_datasets(args, target_type="art"):
    train_set = datasets.ImageFolder(path_dict["photo"])
    test_set = datasets.ImageFolder(path_dict[target_type])
    return train_set, test_set

def get_transforms():
    return transf, transf
    
