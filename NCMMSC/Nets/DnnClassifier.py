import torch
import torch.nn as nn


import torch
import torch.nn as nn
from torch.nn import init


class Classifier(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        in_features = cfg.CLASSIFIER.IN_DIM
        num_layers = cfg.CLASSIFIER.NUM_LAYERS
        hidden_dim = cfg.CLASSIFIER.HIDDEN_DIM
        num_classes = cfg.CLASSIFIER.NUM_CLASSES
        drop_rate = cfg.CLASSIFIER.DROP_RATE
        act = nn.GELU

        self.layers = nn.Sequential()
        in_dim = in_features
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            self.layers.append(act())
            self.layers.append(nn.Dropout(drop_rate))
            in_dim = hidden_dim
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        self.activation = act()
        self.drop = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self.load_weights(cfg)
    
    def forward(self, x):
        x = self.layers(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.classifier(x)

        return x
    
    def load_weights(self, cfg):
        weight_path = cfg.CLASSIFIER.WEIGHTS
        if weight_path != '':
            weights = torch.load(weight_path, map_location='cpu')
            self.load_state_dict(weights)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)


def build_classifier(cfg):
    model = Classifier(cfg)
    device = torch.device(cfg.DEVICE)
    model = model.to(device)
    return model