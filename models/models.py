import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision


def load_pretrained(net, pretrained, device):
    """Load pretrained parameters into net at device."""
    if pretrained:
        if os.path.isfile(pretrained):
            weights = torch.load(pretrained, map_location=device)
            if "module." in list(weights.keys())[0]:
                weights = {k.replace("module.", ""): v for k, v in weights.items()}
            net.load_state_dict(weights)
        else:
            print(f"didn't find {pretrained}. Using default.")
    else:
        print("Using default weights.")
    return net


def ModifiedDenseNet(n_classes, pretrained="densenet.pt", device="cpu"):
    """DenseNet with a modification of the first layer and last layer"""

    # get DenseNet
    net = torch.hub.load("pytorch/vision:v0.10.0", "densenet121", pretrained=True)

    # modify first layer
    net.features.conv0 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    w = (net.features.conv0.weight.sum(1)).unsqueeze(1)
    net.features.conv0.weight = torch.nn.Parameter(w)

    # Modify last layer
    net.classifier = nn.Linear(
        in_features=net.classifier.in_features, out_features=n_classes, bias=True
    )

    # load pretrained parameters
    net = load_pretrained(net, pretrained, device)
    return net


# Model definition InceptionV3
class BasicConv2d(nn.Module):
    """A 2D conv with batch normalization and ReLU. This is the first layer of InceptionV3 which needs to be modified."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


def ModifiedInceptionV3(n_classes, pretrained="inception.pt", device="cpu"):
    """InceptionV3 modified to our need. First and last layer modified."""

    # get InceptionV3
    net = torchvision.models.inception_v3(
        weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False,
    )
    # Modify first layer
    net.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)
    net.AuxLogits = None

    # Modify last layer
    net.fc = nn.Linear(net.fc.in_features, n_classes)

    # load pretrained parameters
    net = load_pretrained(net, pretrained, device)
    return net


def ModifiedResNet(n_classes, pretrained="resnet.pt", device="gpu"):
    """ResNet50 modified to our need. First and last layer adapted."""

    # get ResNet50
    net = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    )

    # Modify first layer
    net.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    w = (net.conv1.weight.sum(1)).unsqueeze(1)
    net.conv1.weight = torch.nn.Parameter(w)

    # Modify last layer
    net.fc = nn.Linear(net.fc.in_features, n_classes)

    # load pretrained parameters
    net = load_pretrained(net, pretrained, device)
    return net


class ensemble(nn.Module):
    def __init__(self, device, pretrained="parameters", avg=False):
        super(ensemble, self).__init__()
        self.n_classes = 46
        self.avg = avg
        self.densenet = ModifiedDenseNet(
            n_classes=self.n_classes,
            pretrained=os.path.join(pretrained, "densenet.pt"),
            device=device,
        )
        self.inception = ModifiedInceptionV3(
            n_classes=self.n_classes,
            pretrained=os.path.join(pretrained, "inception.pt"),
            device=device,
        )
        self.resnet = ModifiedResNet(
            n_classes=self.n_classes,
            pretrained=os.path.join(pretrained, "resnet.pt"),
            device=device,
        )
        # final stack to train
        self.linear = nn.Linear(141, 46)

    def forward(self, data):

        output1 = self.densenet(data)
        if self.training:
            output2, _ = self.inception(data)
        else:
            output2 = self.inception(data)
        output3 = self.resnet(data)

        if self.avg == False:
            concat = torch.cat([output1, output2, output3], dim=1)
            output = self.linear(concat)
            return output
        else:
            return (output1 + output2 + output3) / 3.0
