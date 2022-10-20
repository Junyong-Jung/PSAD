from distutils.command.config import config
from turtle import forward
import torch
from torch import nn
from torchvision.models import vgg16
from pathlib import Path
import plus_variable

class VGG(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

        # placeholder for the gradients
        self.gradients = None
        self.activation = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, target_layer=11):
        result = []
        for i in range(len(nn.ModuleList(self.features))):
            x = self.features[i](x)
            if i == target_layer:
                self.activation = x
                h = x.register_hook(self.activations_hook)
            if i == 2 or i == 5 or i == 8 or i == 11 or i == 14 or i == 17 or i == 20 or i == 23 or i == 26 or i == 29 or i == 32 or i == 35 or i == 38:
                result.append(x)

        return result

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.activation


def make_layers(cfg, use_bias, batch_norm=False):
    layers = []
    in_channels = 3
    outputs = []
    for i in range(len(cfg)):
        if cfg[i] == 'O':
            outputs.append(nn.Sequential(*layers))
        elif cfg[i] == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=1, bias=use_bias)
            torch.nn.init.xavier_uniform_(conv2d.weight)
            if batch_norm and cfg[i + 1] != 'M':
                layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = cfg[i]
    return nn.Sequential(*layers)


def make_arch(idx, cfg, use_bias, batch_norm=False):
    if 'single' in plus_variable.version:
        return VGG(make_layers(cfg[idx], use_bias, batch_norm=batch_norm))
    
    elif 'early' in plus_variable.version:
        return PSAD_earlyfusion_VGG(make_layers(cfg[idx], use_bias, batch_norm=batch_norm))
    
    elif 'late' in plus_variable.version:
        return PSAD_latefusion_VGG(make_layers(cfg[idx], use_bias, batch_norm=batch_norm))



class Vgg16(torch.nn.Module):
    def __init__(self, pretrain):
        super(Vgg16, self).__init__()
        features = list(vgg16('vgg16-397923af.pth').features)

        if not pretrain:
            for ind, f in enumerate(features):
                # nn.init.xavier_normal_(f)
                if type(f) is torch.nn.modules.conv.Conv2d:
                    torch.nn.init.xavier_uniform(f.weight)
                    print("Initialized", ind, f)
                else:
                    print("Bypassed", ind, f)
            # print("Pre-trained Network loaded")
        self.features = nn.ModuleList(features).eval()
        self.output = []

    def forward(self, x):
        output = []
        for i in range(31):
            x = self.features[i](x)
            if i == 1 or i == 4 or i == 6 or i == 9 or i == 11 or i == 13 or i == 16 or i == 18 or i == 20 or i == 23 or i == 25 or i == 27 or i == 30:
                output.append(x)
        return output


def get_networks(config, load_checkpoint=False):
    equal_network_size = config['equal_network_size']
    pretrain = config['pretrain']
    experiment_name = config['experiment_name']
    dataset_name = config['dataset_name']
    normal_class = config['normal_class']
    use_bias = config['use_bias']
    cfg = {
        'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'B': [16, 16, 'M', 16, 128, 'M', 16, 16, 256, 'M', 16, 16, 512, 'M', 16, 16, 512, 'M'],
    }

    if equal_network_size:
        config_type = 'A'
    else:
        config_type = 'B'

    if 'single' in plus_variable.version:
        vgg = Vgg16(pretrain).cuda()
    
    elif 'early' in plus_variable.version:
        vgg = PSAD_earlyfusion_Vgg16(pretrain).cuda()
    
    elif 'late' in plus_variable.version:
        vgg = PSAD_latefusion_Vgg16(pretrain).cuda()

    model = make_arch(config_type, cfg, use_bias, True).cuda()

    for j, item in enumerate(nn.ModuleList(model.features)):
        print('layer : {} {}'.format(j, item))

    if load_checkpoint:
        last_checkpoint = config['last_checkpoint']
        checkpoint_path = "./outputs/{}/{}/checkpoints/".format(experiment_name, dataset_name)
        model.load_state_dict(
            torch.load('{}Cloner_{}_epoch_{}.pth'.format(checkpoint_path, normal_class, last_checkpoint)))
        if not pretrain:
            vgg.load_state_dict(
                torch.load('{}Source_{}_random_vgg.pth'.format(checkpoint_path, normal_class)))
    elif not pretrain:
        checkpoint_path = "./outputs/{}/{}/checkpoints/".format(experiment_name, dataset_name)
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

        torch.save(vgg.state_dict(), '{}Source_{}_random_vgg.pth'.format(checkpoint_path, normal_class))
        print("Source Checkpoint saved!")

    return vgg, model



class PSAD_earlyfusion_VGG(VGG):
    def __init__(self, features):
        super().__init__(features)
        
        num_imgs = plus_variable.num_imgs
            
        self.features[0] = nn.Conv2d(num_imgs*3 ,16,kernel_size=3,stride=1,padding=1,bias=False)


class PSAD_earlyfusion_Vgg16(Vgg16):
    def __init__(self, pretrain):
        super().__init__(pretrain)
        
        num_imgs = plus_variable.num_imgs
        
        weight = self.features[0].weight.clone()
        bias = self.features[0].bias.clone()
        self.features[0] = nn.Conv2d(num_imgs*3, 64, kernel_size=3, stride=1, padding=1)
        
        with torch.no_grad():
            self.features[0].bias = nn.Parameter(bias)
            for i in range(0, num_imgs*3, 3):
                self.features[0].weight[:,i:i+3] = nn.Parameter(weight)

class PSAD_latefusion_VGG(VGG):
    def __init__(self, features):
        super().__init__(features)
        
    def forward(self, x, target_layer=11):
        result = []
        x_list = torch.split(x,3,dim=1)
        y_list = []
        for x in x_list:
        
            for i in range(len(nn.ModuleList(self.features))):
                x = self.features[i](x)
                if i == target_layer:
                    self.activation = x
                    h = x.register_hook(self.activations_hook)
                if i == 2 or i == 5 or i == 8 or i == 11 or i == 14 or i == 17 or i == 20 or i == 23 or i == 26 or i == 29 or i == 32 or i == 35 or i == 38:
                    result.append(x)
        chunk = len(result) // plus_variable.num_imgs
        result = [result[a : a + chunk] for a in range(0,len(result),chunk)]
        for a in zip(*result):
            tmp = torch.stack(a,dim=-1)
            if 'multi_late_mean_pooling' in plus_variable.version:
                result = torch.mean(tmp,dim=-1)
            elif 'multi_late_max_pooling' in plus_variable.version:
                result = torch.max(tmp,dim=-1)[0]
            elif 'multi_late_mean_max_pooling' in plus_variable.version:
                mean_result = torch.mean(tmp,dim=-1)
                max_result = torch.max(tmp,dim=-1)[0]
                result = mean_result + max_result
                
            y_list.append(result)
        return y_list
    
class PSAD_latefusion_Vgg16(Vgg16):
    def __init__(self, pretrain):
        super().__init__(pretrain)
    
    def forward(self, x):
        output = []
        x_list = torch.split(x,3,dim=1)
        y_list = []
        for x in x_list:        
            for i in range(31):
                x = self.features[i](x)
                if i == 1 or i == 4 or i == 6 or i == 9 or i == 11 or i == 13 or i == 16 or i == 18 or i == 20 or i == 23 or i == 25 or i == 27 or i == 30:
                    output.append(x)
        chunk = len(output) // plus_variable.num_imgs
        output = [output[a : a + chunk] for a in range(0,len(output),chunk)]
        for a in zip(*output):
            tmp = torch.stack(a,dim=-1)
            if 'multi_late_mean_pooling' in plus_variable.version:
                result = torch.mean(tmp,dim=-1)
            elif 'multi_late_max_pooling' in plus_variable.version:
                result = torch.max(tmp,dim=-1)[0]
            elif 'multi_late_mean_max_pooling' in plus_variable.version:
                mean_result = torch.mean(tmp,dim=-1)
                max_result = torch.max(tmp,dim=-1)[0]
                result = mean_result + max_result
                
            y_list.append(result)
        return y_list