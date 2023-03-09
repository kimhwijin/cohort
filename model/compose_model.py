from torch import nn
from .image_model import ImageModel
from .feature_model import FeatureModel
from .block import LinearBlock
import torch

class ComposeModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.image_model = ImageModel(config)
        self.feature_model = FeatureModel(config)
        self.out_features = 1
        self.fc = nn.Sequential(
            LinearBlock(self.image_model.out_features+self.feature_model.out_features, 128, activation='relu', batch_norm=config.batch_norm, bias=False),
            nn.Dropout(config.dropout_rate) if hasattr(config.dropout_rate, 'dropout_rate') else nn.Identity(), 
            LinearBlock(128, 64, activation='relu', batch_norm=config.batch_norm, bias=False),
            nn.Dropout(config.dropout_rate) if hasattr(config.dropout_rate, 'dropout_rate') else nn.Identity(), 
            LinearBlock(64, 32, activation='relu', batch_norm=config.batch_norm, bias=False),
            nn.Dropout(config.dropout_rate) if hasattr(config.dropout_rate, 'dropout_rate') else nn.Identity(), 
            LinearBlock(32, self.out_features, activation=None, batch_norm=False, bias=False)
        )

    def forward(self, image, numeric_features, categorical_features):
        z_fea = self.feature_model(numeric_features, categorical_features)
        z_img = self.image_model(image)
        logit = self.fc(torch.concat([z_fea, z_img], dim=-1))
        return logit
