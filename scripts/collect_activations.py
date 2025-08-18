import torch
import argparse
import numpy as np
from tqdm import tqdm

from src.collector import ActivationsCollector, store_activations, checkpoint_activations
from src.utils import setup_loader, load_inceptionV1, imagenet_preprocess, compute_free_cuda_mem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="__download__")
    parser.add_argument("--out-dir", default="latent_data")
    parser.add_argument("--layer-names", nargs="+", default=["inception3a", "inception4b", "inception5a"])
    parser.add_argument("--samples-per-image", type=int, default=512)
    args = parser.parse_args()

    preprocess = imagenet_preprocess()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = load_inceptionV1(device)
    loader = setup_loader(args.img_dir, preprocess, batch_size=8)

    collector = ActivationsCollector(model, args.layer_names)

    all_activations = {}
    gpu_cache = {}

    with torch.no_grad():
        for x, _ in tqdm(loader):
            x = x.to(device)
            model(x)

            for k, v in collector.activations.items():
                gpu_cache[k] = v if k not in gpu_cache else torch.cat([gpu_cache[k], v])

            if device.type == "cuda" and compute_free_cuda_mem() < 0.3:
                checkpoint_activations(gpu_cache, all_activations)

    checkpoint_activations(gpu_cache, all_activations)
    store_activations(
        all_activations,
        args.out_dir,
        samples_per_image=args.samples_per_image
    )

if __name__ == "__main__":
    main()