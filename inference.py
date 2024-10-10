import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torchvision
import torch
from models.models import ModifiedResNet
import argparse

def plot_kws(data, log=True):
    if log == True:
        norm=LogNorm()
    else:
        norm=None
    plt.imshow(data, norm=norm,origin='lower')
    plt.axis('off')

def preprocess_data(npy):
    transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((180,180)),
        torch.nn.ReLU(inplace=True),
        torchvision.transforms.Lambda(lambda x: torch.log(x+1.0)),
        torchvision.transforms.Lambda(lambda x: x/torch.max(x) if torch.max(x)>0 else x)
    ])

    data = np.load(npy)
    t_data = transforms(data)
    return t_data

def main(fname, plot=False):

    # Load data
    t_data = preprocess_data(fname)

    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = ModifiedResNet(n_classes=46, pretrained="./parameters/resnet.pt", device=device)
    net.eval()
    net.to(device)

    # inference
    with torch.no_grad():
        output = net(t_data.unsqueeze(0).float().to(device))
    probabilities = torch.nn.functional.softmax(output, dim=1).squeeze().numpy()
    positional_index = np.argsort(probabilities)[::-1]

    # optional plotting
    if plot:
        plt.figure(figsize=(10,6))
        plot_kws(t_data.squeeze(),log=False)
        plt.tight_layout()
        plt.show()

    return probabilities, positional_index

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename")
    parser.add_argument("-p", "--plot")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    probabilities, positional_index = main(fname = args.filename, plot=args.plot)
    with open("sas_models.txt","r") as f:
        lines = f.readlines()

    for j, i in enumerate(positional_index):
        print(f"{j+1}: {lines[i].strip()} (prob: {probabilities[i]:.5f})")


