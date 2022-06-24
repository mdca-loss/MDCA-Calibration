import torch
from torchvision.models.resnet import ResNet, BasicBlock
from torch.hub import load_state_dict_from_url

resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'

def resnet18_pacs(num_classes, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    state_dict = load_state_dict_from_url(resnet18_url, progress=True)
    model.load_state_dict(state_dict, strict=False)
    for params in model.parameters():
        params.requires_grad = False
    model.fc = torch.nn.Linear(BasicBlock.expansion * 512, num_classes)
    return model