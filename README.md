[![Code-Generator](https://badgen.net/badge/Template%20by/Code-Generator/ee4c2c?labelColor=eaa700)](https://github.com/pytorch-ignite/code-generator)

# Image Classification Template by Code-Generator

This is the image classification template by Code-Generator using `resnet18` model and `cifar10` dataset from TorchVision and training is powered by PyTorch and PyTorch-Ignite.

## Modification
The original codes are modified to use `ImageFolder` from `torchvision`. See `data.py`.

For more information visit:
* [Pytorch Ignite](https://github.com/pytorch/ignite)
* [Code Generator](https://github.com/pytorch-ignite/code-generator)


## Getting Started

Install the dependencies with `pip`:

```sh
pip install -r requirements.txt --progress-bar off -U
```

## Training

### 1 GPU Training

```sh
python main.py
```
