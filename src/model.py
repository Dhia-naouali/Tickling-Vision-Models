import torch
from torchvision import models

def load_inceptionV1(device=None):
    model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    model.eval().to(device)
    return model

def imagenet_preprocess():
    weights = models.GoogLeNet_Weights.DEFAULT
    return weights.transforms()