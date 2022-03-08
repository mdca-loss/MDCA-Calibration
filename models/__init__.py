from .resnet import resnet20, resnet32, resnet56, resnet110
from .mendley_networks import resnet50_mendley
from .resnet_pacs import resnet18_pacs
from .resnet_mnist import resnet20_mnist
from .resnet_imagenet import resnet34, resnet50

model_dict = {
    # resnet models can be used for cifar10/100, svhn
    # mendley models only to be used for mendley datasets

    "resnet20" : resnet20,
    "resnet32" : resnet32,
    "resnet56" : resnet56,
    "resnet110" : resnet110,

    "resnet50_mendley" : resnet50_mendley,

    "resnet_pacs" : resnet18_pacs,

    "resnet_mnist" : resnet20_mnist,

    "resnet34_imagenet" : resnet34,
    "resnet50_imagenet" : resnet50
}