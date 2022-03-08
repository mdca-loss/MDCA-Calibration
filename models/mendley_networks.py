import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import numpy as np


class CNN(nn.Module):
    # input: input_shape:	[num_of_filters, kernel_size] (e.g. [256, 25])
    def __init__(self, 
                 num_classes=2, 
                 feature='Alex', 
                 feature_shape=(256,25,25), 
                 pretrained=True, 
                 requires_grad=False, **kwargs):         
        
        super(CNN, self).__init__()

        
        # Feature Extraction
        if(feature=='Alex'):            
            self.ft_ext = models.alexnet(pretrained=pretrained) 
            self.ft_ext_modules = list(self.ft_ext.children())[:-1]
        elif(feature=='Res'):
            self.ft_ext = models.resnet50(pretrained=pretrained) 
            self.ft_ext_modules=list(self.ft_ext.children())[:-2]
        elif(feature=='Squeeze'):
            self.ft_ext = models.squeezenet1_1(pretrained=pretrained) 
            self.ft_ext_modules=list(self.ft_ext.children())[:-1]            
        elif(feature=='Dense'):
            self.ft_ext = models.densenet161(pretrained=pretrained) 
            self.ft_ext_modules=list(self.ft_ext.children())[:-1]
            
        self.ft_ext=nn.Sequential(*self.ft_ext_modules)                
        for p in self.ft_ext.parameters():
            p.requires_grad = requires_grad

        # Classifier                   
        if(feature=='Res'):
            conv1_output_features = int(feature_shape[0]/4)
        else:
            conv1_output_features = int(feature_shape[0]/2)
        fc_1_input_features = conv1_output_features*int(feature_shape[1]/2)*int(feature_shape[2]/2)
        # print("fc1_in_shape:", fc_1_input_features)
        fc1_output_features = 256
        fc2_output_features = 64         
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=feature_shape[0],        
                out_channels=conv1_output_features,       
                kernel_size=1,          
            ),
            nn.BatchNorm2d(conv1_output_features),
            nn.ReLU(),                  
            nn.MaxPool2d(kernel_size=2)
        )                    
        
        self.fc1 = nn.Sequential(
             nn.Linear(fc_1_input_features, fc1_output_features),
             nn.BatchNorm1d(fc1_output_features),            
             nn.ReLU()
         )

        self.fc2 = nn.Sequential(
             nn.Linear(fc1_output_features, fc2_output_features),
             nn.BatchNorm1d(fc2_output_features),
             nn.ReLU()
         )
        
        self.out = nn.Linear(fc2_output_features, num_classes)

    def forward(self, x, drop_prob=0.5):
        ft = self.ft_ext(x)           
        x = self.conv1(ft)        
        x = x.view(x.size(0), -1) 

        # print(x.shape)          
        x = self.fc1(x)
        x = nn.Dropout(drop_prob)(x)
        x = self.fc2(x)
        x = nn.Dropout(drop_prob)(x)        
        prob = self.out(x) 
        return prob

def resnet50_mendley(**kwargs):
    return CNN(**kwargs, feature='Res', feature_shape=[2048, 26, 26])

def alexnet_mendley(**kwargs):
    return CNN(**kwargs, feature='Alex', feature_shape=[256, 6, 6])