import os
import cv2
import torch

from torchvision import models
from torch.utils.data import DataLoader, Subset

from torchvision.datasets import ImageFolder
from datasets import load_dataset


def compute_free_cuda_mem():
    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved(0)
    return (total - reserved) / total


def load_inceptionV1(device=None):
    model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
    model.eval().to(device)
    return model

def imagenet_preprocess():
    weights = models.GoogLeNet_Weights.DEFAULT
    return weights.transforms()


def setup_loader(img_dir, preprocess, batch_size=8, num_samples=5e3):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": os.cpu_count()
    }
    if img_dir == "__download__":
        def collate_fn(batch):
            images = [preprocess(x["image"].convert("RGB")) for x in batch]
            labels = [x["label"] for x in batch]
            return torch.stack(images), torch.tensor(labels)
        dataset = load_dataset(
            "timm/mini-imagenet", 
            split="validation"
        )
        classes = [0, 3, 5, 6, 7, 9, 21, 27, 29, 86]
        return DataLoader(dataset.filter(lambda x: x["label"] in classes), **kwargs, collate_fn=collate_fn)

    dataset = ImageFolder(img_dir, transform=preprocess)
    indices = list(range(min(num_samples, len(dataset))))
    return DataLoader(Subset(dataset, indices), **kwargs)