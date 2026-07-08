import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

            
def ResNet18():
 weights = ResNet18_Weights.DEFAULT
 model = resnet18(weights=weights)

 model.fc = nn.Linear(512,9)

 for param in model.parameters():
    param.requires_grad = False
    
 for param in model.fc.parameters():
    param.requires_grad = True
    
 for param in model.layer4.parameters():
    param.requires_grad=False  
 
 for param in model.layer3.parameters():
    param.requires_grad=False
 return  model

