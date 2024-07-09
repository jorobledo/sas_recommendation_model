import torch.nn as nn
import os 
import torch
import torch.nn.functional as F
import torchvision

def load_pretrained(net, pretrained, device):
    if pretrained:
            if os.path.isfile(pretrained):
                weights = torch.load(pretrained,
                                    map_location=device)
                if 'module.' in list(weights.keys())[0]:
                    weights = {k.replace('module.',''):v for k, v in weights.items()}
                net.load_state_dict(weights)
            else:
                print(f"didn't find {pretrained}. Using default.")
    else:
        print('Using default weights.')
    return net

def ModifiedDenseNet(path_data, pretrained='densenet.pt', device='cpu'):
    net = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    net.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    w = (net.features.conv0.weight.sum(1)).unsqueeze(1)
    net.features.conv0.weight = torch.nn.Parameter(w)
    num_ftrs = net.classifier.in_features
    classes = sorted(entry.name for entry in os.scandir(path_data) if entry.is_dir())
    net.classifier = nn.Linear(in_features=num_ftrs, out_features=len(classes), bias=True)
    print(net)
    net = load_pretrained(net, pretrained, device)
    return net

# Model definition
class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

def ModifiedInceptionV3(path_data, pretrained='inception.pt', device='cpu'):   
        net = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1,
                                             transform_input=False)
        net.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)
        net.AuxLogits = None
        num_ftrs = net.fc.in_features
        classes = sorted(entry.name for entry in os.scandir(path_data) if entry.is_dir())
        net.fc = nn.Linear(num_ftrs, len(classes))
        print(net)
        net = load_pretrained(net, pretrained, device)
        return net

def ModifiedResNet(path_data, pretrained='resnet.pt', device='gpu'):
    net = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
    net.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    w = (net.conv1.weight.sum(1)).unsqueeze(1)
    net.conv1.weight = torch.nn.Parameter(w)
    num_ftrs = net.fc.in_features
    classes = sorted(entry.name for entry in os.scandir(path_data) if entry.is_dir())
    net.fc = nn.Linear(num_ftrs, len(classes))
    print(net)
    net = load_pretrained(net, pretrained, device)
    return net