import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import alexnet
import plus_variable
import config as c
from freia_funcs import permute_layer, glow_coupling_layer, F_fully_connected, ReversibleGraphNet, OutputNode, \
    InputNode, Node

WEIGHT_DIR = './weights'
MODEL_DIR = './models'


def nf_head(input_dim=c.n_feat):
    nodes = list()
    nodes.append(InputNode(input_dim, name='input'))
    for k in range(c.n_coupling_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                          {'clamp': c.clamp_alpha, 'F_class': F_fully_connected,
                           'F_args': {'internal_size': c.fc_internal, 'dropout': c.dropout}},
                          name=F'fc_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    coder = ReversibleGraphNet(nodes)
    return coder


class DifferNet(nn.Module):
    def __init__(self):
        super(DifferNet, self).__init__()
        self.feature_extractor = alexnet(pretrained=True)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False        
        self.nf = nf_head()

    def forward(self, x):
        y_cat = list()

        for s in range(c.n_scales):
            x_scaled = F.interpolate(x, size=c.img_size[0] // (2 ** s)) if s > 0 else x
            feat_s = self.feature_extractor.features(x_scaled)
            y_cat.append(torch.mean(feat_s, dim=(2, 3)))

        y = torch.cat(y_cat, dim=1)
        z = self.nf(y)
        return z

class PSAD_earlyfusion_DifferNet(DifferNet):
    def __init__(self):
        super().__init__()
        weight = self.feature_extractor.features[0].weight.clone()
        bias = self.feature_extractor.features[0].bias.clone()
        self.feature_extractor.features[0] = nn.Conv2d(12, 64, kernel_size=11, stride=4, padding=2)
        
        with torch.no_grad():
            self.feature_extractor.features[0].bias = nn.Parameter(bias)
            self.feature_extractor.features[0].weight[:,:3] = nn.Parameter(weight)
            self.feature_extractor.features[0].weight[:,3:6] = nn.Parameter(weight)
            self.feature_extractor.features[0].weight[:,6:9] = nn.Parameter(weight)
            self.feature_extractor.features[0].weight[:,9:12] = nn.Parameter(weight)

class PSAD_latefusion_DifferNet(DifferNet):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        
        x_list = torch.split(x,3,dim=1)
        y_list = []
        for x in x_list:
            
            y_cat = list()

            for s in range(c.n_scales):
                x_scaled = F.interpolate(x, size=c.img_size[0] // (2 ** s)) if s > 0 else x
                feat_s = self.feature_extractor.features(x_scaled)
                y_cat.append(torch.mean(feat_s, dim=(2, 3)))

            y = torch.cat(y_cat, dim=1)
            y_list.append(y)
            
        y = torch.stack(y_list, dim=-1)

        if 'multi_late_mean_pooling' in plus_variable.version:
            y = torch.mean(y, dim=-1)
        elif 'multi_late_max_pooling' in plus_variable.version:
            y = torch.max(y, dim=-1)[0]
        elif 'multi_late_mean_max_pooling' in plus_variable.version:
            y_mean = torch.mean(y,dim=-1)
            y_max = torch.max(y,dim=-1)[0]
            y = y_mean + y_max
        z = self.nf(y)
        return z


def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR, filename))


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    model = torch.load(path)
    return model


def save_weights(model, filename):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, filename))


def load_weights(model, filename):
    path = os.path.join(WEIGHT_DIR, filename)
    model.load_state_dict(torch.load(path))
    return model
