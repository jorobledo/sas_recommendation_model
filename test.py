import concurrent.futures
import torchvision
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse

from models.models import (
    ModifiedDenseNet,
    ModifiedInceptionV3,
    ModifiedResNet,
    ensemble,
)

from utils.load_hdf import H5Dataset
from utils.cnn_utils import accuracy_top


parser = argparse.ArgumentParser(
    prog='test',
    description='Calculate the average batch accuracy of each model'
)

parser.add_argument('--batch_size', type=int, default=10, help="Batch size for the data loader. The larger the size, the more accurate the accuracy but it gets slower.")
parser.add_argument('--avg_batches', type=int, default=5, help="Number of batcher over which to calculate the average (smaller number runs faster, but less accurate).")
parser.add_argument('--num_workers', type=int, default=2, help="Number of workers for the pytorch dataloader." )
args = parser.parse_args()

batch_size = args.batch_size
avg_batches = args.avg_batches
num_workers_loader = args.num_workers

n_classes = 46 # set by the problem we are presenting in our paper.

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

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers_loader,
    shuffle=True,
)
print(f"amount of test minibatches: {len(val_dataloader)}")

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


def evaluate_batch(net, device, images, labels):
    with torch.no_grad():
        output = net(images.to(device))
        _, pred = torch.max(output, dim=1)

    acc_val = accuracy_top(output, labels, topk=(1, 3, 5))
    return acc_val[0].item(), acc_val[1].item(), acc_val[2].item()

def evaluate_net_parallel(neti, net, device, val_dataloader, avg_batches):
    net.to(device)
    net.eval()
    print(neti)

    accuracies = []
    accuracies3 = []
    accuracies5 = []
    it = iter(val_dataloader)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in tqdm(range(avg_batches)):
            images, labels = next(it)
            futures.append(executor.submit(evaluate_batch, net, device, images, labels))

            # Just average the first avg_batches batches to speed up
            if i == avg_batches:
                break

        for future in concurrent.futures.as_completed(futures):
            acc1, acc3, acc5 = future.result()
            accuracies.append(acc1)
            accuracies3.append(acc3)
            accuracies5.append(acc5)

    print(
        f"{neti} Top 1 Avg. accuracy: {100*np.mean(accuracies):.2f} (avg.accuracy over {avg_batches} batches)"
    )
    print(
        f"{neti} Top 3 Avg. accuracy: {100*np.mean(accuracies3):.2f} (avg.accuracy over {avg_batches} batches)"
    )
    print(
        f"{neti} Top 5 Avg. accuracy: {100*np.mean(accuracies5):.2f} (avg.accuracy over {avg_batches} batches)"
    )

def parallel_evaluation(nets, device, val_dataloader, avg_batches):
    for neti in nets:
        net = nets[neti]
        evaluate_net_parallel(neti, net, device, val_dataloader, avg_batches)
        
parallel_evaluation(nets, device, val_dataloader, avg_batches)