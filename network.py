from typing import Tuple
from timm.models.helpers import load_state_dict
import torch
import torch.nn as nn
import timm
# import segmentation_models_pytorch as smp
from torch.nn.modules import transformer

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        print('Weight Init')


class SAMSUNG_RegNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = timm.create_model(args.encoder_name, pretrained=True,
                                    drop_path_rate=args.drop_path_rate,
                                    )
        num_head = self.encoder.head.fc.in_features
        self.encoder.head.fc = nn.Linear(num_head, 3)
    
    def forward(self, x):
        x = self.encoder(x)
        return x

class SAMSUNG_RegNet_test(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=True,
                                    drop_path_rate=0,
                                    )
        num_head = self.encoder.head.fc.in_features
        self.encoder.head.fc = nn.Linear(num_head, 3)
    
    def forward(self, x):
        x = self.encoder(x)
        return x
