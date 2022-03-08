from .cifar10 import get_train_valid_test_loader as cifar10loader
from .cifar10 import get_datasets as cifar10datasets
from .cifar10 import get_transforms as cifar10transforms

from .cifar100 import get_train_valid_test_loader as cifar100loader
from .cifar100 import get_datasets as cifar100datasets
from .cifar100 import get_transforms as cifar100transforms

from .svhn import get_train_valid_test_loader as svhnloader
from .svhn import get_datasets as svhndatasets
from .svhn import get_transforms as svhntransforms

from .mendley import get_train_valid_test_loader as mendleyloader
from .mendley import get_datasets as mendleydatasets
from .mendley import get_transforms as mendleytransforms

from .pacs import get_train_valid_test_loader as pacs_corruptedloader
from .pacs import get_datasets as pacs_corrupteddatasets
from .pacs import get_transforms as pacs_corruptedtransforms

from .mnist import get_train_valid_test_loader as mnistloader
from .mnist import get_datasets as mnistdatasets
from .mnist import get_transforms as mnisttransforms

from .imagenet import get_train_valid_test_loader as imagenetloader
from .imagenet import get_datasets as imagenetdatasets
from .imagenet import get_transforms as imagenettransforms

from .imbalanced_cifar import get_transforms as imbalanced_cifartransforms
from .imbalanced_cifar import get_train_valid_test_loader as imbalanced_cifarloader

dataloader_dict = {
    "cifar10" : cifar10loader,
    "cifar100" : cifar100loader,
    "svhn" : svhnloader,
    "mendley" : mendleyloader,
    "pacs" : pacs_corruptedloader,
    "mnist" : mnistloader,
    "imagenet" : imagenetloader,
    "im_cifar10" : imbalanced_cifarloader
}

corrupted_dataloader_dict = {
    "pacs" : pacs_corruptedloader,
    "mnist" : mnistloader,
}

dataset_dict = {
    "cifar10" : cifar10datasets,
    "cifar100" : cifar100datasets,
    "svhn" : svhndatasets,
    "mendley" : mendleydatasets,
    "pacs" : pacs_corrupteddatasets,
    "mnist" : mnistdatasets,
    "imagenet" : imagenetdatasets,
}

corrupted_dataset_dict = {
    "pacs" : pacs_corrupteddatasets,
    "mnist" : mnistdatasets
}

dataset_nclasses_dict = {
    "cifar10" : 10,
    "cifar100" : 100,
    "svhn" : 10,
    "mendley" : 2,
    "pacs" : 7,
    "mnist" : 10,
    "imagenet" : 200,
    "im_cifar10" : 10
}

dataset_classname_dict = {
    "cifar10" : ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],

    "cifar100" : ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
                'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud',
                'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
                'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 
                'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 
                'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
                'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
                'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
                'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
                'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
                'whale', 'willow_tree', 'wolf', 'woman', 'worm'],
                
    "svhn" : [f"{i}" for i in range(10)],

    "mendley" : ["Normal", "Abnormal"],

    "pacs" : ['Dog', 'Elephant', 'Giraffe', 'Guitar', 'Horse', 'House', 'Person'],

    "mnist" : [f"{i}" for i in range(10)],

    "imagenet" : [f"{i}" for i in range(200)],

    "im_cifar10" : ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
}

dataset_transform_dict = {
    "cifar10" : cifar10transforms,
    "cifar100" : cifar100transforms,
    "svhn" : svhntransforms,
    "mendley" : mendleytransforms,
    "pacs" : pacs_corruptedtransforms,
    "mnist" : mnisttransforms,
    "imagenet" : imagenettransforms,
}
