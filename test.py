import torchvision
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from models.models import (
    ModifiedDenseNet,
    ModifiedInceptionV3,
    ModifiedResNet,
    ensemble,
)

from utils.load_hdf import H5Dataset
from utils.cnn_utils import accuracy_top

imsize_w = 180
imsize_h = 180
num_workers_loader = 2
n_classes = 46

transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((imsize_w, imsize_h), antialias=True),
        torch.nn.ReLU(inplace=True),  # remove negative values if any
        torchvision.transforms.Lambda(lambda x: torch.log(x + 1.0)),
        torchvision.transforms.Lambda(
            lambda x: x / torch.max(x) if torch.max(x) > 0 else x
        ),
    ]
)

val_dataset = H5Dataset("../data/test.h5", transforms=transforms)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=1000,
    num_workers=num_workers_loader,
    shuffle=True,
)
print(f"amount of test minibatches: {len(val_dataloader)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nets = {
    "densenet": ModifiedDenseNet(
        n_classes=n_classes, pretrained="parameters/densenet.pt", device=device
    ),
    "inception": ModifiedInceptionV3(
        n_classes=n_classes, pretrained="parameters/inception.pt", device=device
    ),
    "resnet": ModifiedResNet(
        n_classes, pretrained="parameters/resnet.pt", device=device
    ),
    "ensemble": ensemble(device=device, pretrained="parameters", avg=True),
}


for neti in nets:
    net = nets[neti]
    net.to(device)
    net.eval()
    print(neti)

    avg_batches = 4
    accuracies = []
    it = iter(val_dataloader)
    for i in tqdm(range(avg_batches)):
        images, labels = next(it)

        # Forward pass of batch
        with torch.no_grad():
            output = net(images.to(device))
            _, pred = torch.max(output, dim=1)

        acc_val = accuracy_top(output, labels, topk=(1, 3, 5))
        accuracies.append(acc_val[0].item())
        with open(f"report_{neti}.txt", "a") as f:
            print([a.item() for a in acc_val], file=f)

        # just average the first vg_batche batches to speadup
        if i == avg_batches:
            break

    print(
        f"{neti}: {100*np.mean(accuracies):.2f} (avg.accuracy over {avg_batches} batches)"
    )
