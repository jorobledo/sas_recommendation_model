# Small Angle Scattering recommendation system based on CNNs

This repository is related to the [paper](https://www.nature.com/articles/s41598-024-65712-y) "Learning from virtual experiments to assist users of Small Angle Neutron Scattering n model selection"

The authors of the paper are [Jose Robledo](https://github.com/jorobledo), [Peter Willendrup](https://orbit.dtu.dk/en/persons/peter-kj%C3%A6r-willendrup), [Henrich Frielinghaus](https://www.fz-juelich.de/profile/frielinghaus_h), and [Klaus Lieutenant](https://www.fz-juelich.de/profile/lieutenant_k), and the code was written by Jose Robledo.

**Note** current software works well with PyTorch 2.2.1 .

If you found this repo useful, please [cite us](#citation). 


## Prerequisites
- python 3
- CPU or NVIDIA GPU

## Installation
- Clone this repo:
```bash
git clone https://github.com/jorobledo
cd SAS_recommendation_model
```
- Install [PyTorch](http://pytorch.org) and other dependencies:
```bash
pip install -r requirements.txt
```
or 
```bash
conda env create -f environment.yml
```

## Download dataset
The full dataset of virtual experiments at the KWS-1 beamline of the FRM-II reactor in Garching used in our paper is published in Zenodo and can be [downloaded here](https://zenodo.org/records/10119316).

- Download the KWS-1 virtual experiments dataset from zenodo by running:
```bash
bash ./utils/get_data.sh
```
This will download the full dataset in the `data` folder. Make sure that the script is run from the current folder for a correct download path. If not, modify the download path in `./utils/det_data.sh` to the desired location. 

**note**: The dataset is large, therefore this might take a while.

## Instructions


## Citation
If you use this code for your research, please cite our paper.

[1] J.I. Robledo, H. Frielinghaus, P. Willendrup and K. Lieutenant, *Learning from virtual experiments to assist users of Small Angle Neutron Scattering in model selection*, Scientific Reports 14 (2024), 14996. doi:https://doi.org/10.1038/s41598-024-65712-y.

BibTex:
```
@article{robledo2024learning,
  title={Learning from virtual experiments to assist users of Small Angle Neutron Scattering in model selection},
  author={Robledo, Jos{\'e} Ignacio and Frielinghaus, Henrich and Willendrup, Peter and Lieutenant, Klaus},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={14996},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
