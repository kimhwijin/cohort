from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def get_activation_layer(activation, **kwargs):
    if activation == 'relu':
        return nn.ReLU(**kwargs)
    elif activation == 'relu6':
        return nn.ReLU6(**kwargs)
    elif activation == 'sigmoid':
        return nn.Sigmoid(**kwargs)
    elif activation == 'softmax':
        return nn.Softmax(**kwargs)
    elif activation == None:
        return nn.Identity()
    

def get_backbone_model(name, out_features):
    if name == 'resnet18':
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Linear(backbone.fc.in_features, out_features)
        return backbone
    


