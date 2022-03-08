# inspired from https://github.com/GB-TonyLiang/DCA/blob/master/dataset.py
from torch.utils.data import Dataset
import numpy as np
import cv2
import csv
import random
from torch.utils import data
from PIL import Image

from torchvision import transforms

def read_csv(data_file_path):
    data = []
    with open(data_file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        data = np.asarray(data)
    return data

transform_train = transforms.Compose([
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

class MendleyDataset(Dataset):
    def __init__(self, 
                 label_file,
                 resize=True,
                 augmentation=True,
                 transforms=None):
        
        self.resize = resize        
        self.augmentation = augmentation
        self.files = read_csv(label_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        temp = self.files[idx]
        full_path = temp[0]
        label = temp[1]
        
        im = cv2.imread(full_path)/255. 

        if(self.resize):
            im = cv2.resize(im,(832,832))        
                
        if(self.augmentation):
            # apply random flip
            flip = np.random.randint(4, size=3)
            if(flip[0]):
                im = cv2.flip(im,0) # flip horizontally
            elif(flip[1]):
                im = cv2.flip(im,1) # flip vertically
            elif(flip[2]):
                im = cv2.flip(im,-1) # flip both horizontally and vertically

            # random rotate
            rotate = np.random.randint(4)
            (h, w) = im.shape[:2]
            # calculate the center of the image
            center = (w / 2, h / 2)
            angle90 = 90
            angle180 = 180
            angle270 = 270
            scale = 1.0

            # Perform the counter clockwise rotation holding at the center
            if(rotate==0):
                M = cv2.getRotationMatrix2D(center, angle90, scale)           
                im = cv2.warpAffine(im, M, (h, w))
            elif(rotate==1):
                M = cv2.getRotationMatrix2D(center, angle180, scale)           
                im = cv2.warpAffine(im, M, (h, w))
            elif(rotate==2):
                M = cv2.getRotationMatrix2D(center, angle270, scale)           
                im = cv2.warpAffine(im, M, (h, w))
        
        # print(im.shape)
        # print(im)
        image = Image.fromarray((im * 255).astype(np.uint8))

        if self.transforms:
            image = self.transforms(image)
        
        return image, int(label)


def get_train_valid_test_loader(args):
    train_set = MendleyDataset(label_file="data/mendley/mendley_train.csv", resize=True, augmentation=True, transforms=transform_train)
    val_set = MendleyDataset(label_file="data/mendley/mendley_train.csv", resize=True, augmentation=True, transforms=transform_test)

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

    test_set = MendleyDataset(label_file="data/mendley/mendley_test.csv", resize=True, augmentation=False, transforms=transform_test)
    test_loader = data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    return train_loader, val_loader, test_loader

def get_datasets(args):
    trainset = MendleyDataset(label_file="data/mendley/mendley_train.csv", resize=True, augmentation=False)
    testset = MendleyDataset(label_file="data/mendley/mendley_test.csv", resize=True, augmentation=False)
    return trainset, testset

def get_transforms():
    return transform_train, transform_test