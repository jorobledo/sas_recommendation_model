import torchvision
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import torch.nn.functional as F

from models.models import (
    ModifiedDenseNet,
    ModifiedInceptionV3,
    ModifiedResNet,
    ensemble,
)

from utils.load_hdf import H5Dataset
from utils.cnn_utils import accuracy_top


batch_size = 10
avg_batches = 5
num_workers = 2
n_classes = 46

transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((180, 180), antialias=True),
        torch.nn.ReLU(inplace=True),  # remove negative values if any
        torchvision.transforms.Lambda(lambda x: torch.log(x + 1.0)),
        torchvision.transforms.Lambda(
            lambda x: x / torch.max(x) if torch.max(x) > 0 else x
        ),
    ]
)

val_dataset = H5Dataset("./data/val.h5", transforms=transforms)

print(val_dataset[0][0].shape)
print(len(val_dataset))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nets = {
    "densenet": ModifiedDenseNet(
        n_classes=n_classes, pretrained="./parameters/densenet.pt", device=device
    ),
    "inception": ModifiedInceptionV3(
        n_classes=n_classes, pretrained="./parameters/inception.pt", device=device
    ),
    "resnet": ModifiedResNet(
        n_classes, pretrained="./parameters/resnet.pt", device=device
    ),
    "ensemble": ensemble(device=device, pretrained="./parameters", avg=True),
}

neti = "resnet"
net = nets[neti]
net.to(device)
net.eval()

num_images = int(len(val_dataset) * 0.01)
print(num_images)
for i in tqdm(range(num_images)):
    images, label = val_dataset[i]
    images = images.unsqueeze(0)

    with torch.no_grad():
        output = net(images.to(device))
        probs = F.softmax(output, dim=1)
        val, pred = torch.max(probs, dim=1)
        print(val)
        print(pred, label)

    
    if i>2:
        break

