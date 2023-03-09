from torch import nn
from .utils import get_activation_layer

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', batch_norm=False, **kwargs):
        super().__init__()
        self.in_features = in_features; self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, **kwargs)
        self.act = get_activation_layer(activation)
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_features)
        else:
            self.bn = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.linear(x)))


