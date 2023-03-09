from torch import nn
from .block import LinearBlock
import torch

class FeatureModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # variables
        self.config = config
        self.n_numeric = len(self.config.NUMERIC_FEATURES)

        self.n_categorical = len(self.config.CATEGORICAL_FEATURES)
        self.total_categorical_count = 0
        for cats in self.config.ONE_HOT_CAT_FEATURES: self.total_categorical_count += len(cats)

        # layers
        self.out_features = 256
        self.emb = nn.Embedding(self.total_categorical_count, 16)
        self.fc_categorical = nn.Sequential(
            LinearBlock(self.n_categorical*16, 64, activation='relu', batch_norm=config.batch_norm),
            )
        self.fc_numeric = nn.Sequential(
            LinearBlock(self.n_numeric, 64, activation='relu', batch_norm=config.batch_norm),
            LinearBlock(64, 128, activation='relu', batch_norm=config.batch_norm),
            )
        self.fc = nn.Sequential(
            LinearBlock(128+64, 128, activation='relu', batch_norm=config.batch_norm),
            LinearBlock(128, self.out_features, activation='relu', batch_norm=config.batch_norm),
        )
    

    def forward(self, numeric_features, categorical_features):
        numeric = self.fc_numeric(numeric_features)
        categorical = self.fc_categorical(self.emb(categorical_features).view(categorical_features.shape[0], -1))
        x = torch.concat([numeric, categorical], dim=-1)
        return self.fc(x)
