from torch import nn
from .utils import get_backbone_model, get_activation_layer

class ImageModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        self.backbone = get_backbone_model(config.backbone_name, out_features=256)

        self.out_features = 256
        self.fc = nn.Sequential(
            nn.BatchNorm1d(256),
            get_activation_layer('relu'),
            nn.Linear(256, self.out_features),
        )
    
    def forward(self, x):
        return self.fc(self.backbone(x))
    
        