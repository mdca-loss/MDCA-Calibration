import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
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

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, phase="train", imbalance_ratio=10, root = 'data', imb_type='exp', transform=None):
        train = True if phase == "train" else False
        super(IMBALANCECIFAR10, self).__init__(root, train, transform=None, target_transform=None, download=True)
        self.train = train
        if self.train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio)
            self.gen_imbalanced_data(img_num_list)
            print(img_num_list)

        self.labels = self.targets
        self.transform = transform

        # import pdb; pdb.set_trace()

        print("{} Mode: Contain {} images".format(phase, len(self.data)))

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict


    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label
    
    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

def get_train_valid_test_loader(args):
    train_set = IMBALANCECIFAR10(phase='train', imbalance_ratio=args.imbalance, root='./data', transform=transform_train)

    # create a val set from training set
    # idxs = list(range(len(train_set)))
    # random.seed(args.seed)
    # random.shuffle(idxs)
    # split = int(0.1 * len(idxs))
    # train_idxs, valid_idxs = idxs[split:], idxs[:split]

    # train_sampler = data.SubsetRandomSampler(idxs)
    # val_sampler = data.SubsetRandomSampler(valid_idxs)

    train_loader = data.DataLoader(train_set, batch_size=args.train_batch_size, num_workers=args.workers, shuffle=True)
    # val_loader = data.DataLoader(train_set, batch_size=args.test_batch_size, num_workers=args.workers, sampler=val_sampler, drop_last=False)

    test_set = IMBALANCECIFAR10(phase='test', transform=transform_test)
    test_loader = data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    return train_loader, None, test_loader

def get_datasets(args):
    trainset = IMBALANCECIFAR10(phase='train', root='./data', transform=None)
    testset = IMBALANCECIFAR10(phase='test', root='./data', transform=None)
    return trainset, testset

def get_transforms():
    return transform_train, transform_test