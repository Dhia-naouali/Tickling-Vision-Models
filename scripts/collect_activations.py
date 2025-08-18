import torch
import argparse
import numpy as np
from tqdm import tqdm

from src.collector import ActivationsCollector, store_activations
from src.utils import setup_loader, load_inceptionV1, imagenet_preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="__download__")
    parser.add_argument("--out-dir", default="latent_data")
    parser.add_argument("--layers", nargs="+", default=["..."])
    parser.add_argument("--samples-per-image", type=int, default=512)
    args = parser.parse_args()

    preprocess = imagenet_preprocess()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = load_inceptionV1(device)
    loader = setup_loader(args.images_dir, preprocess, batch_size=8)

    layer_names = []

    collector = ActivationsCollector(model, layer_names)

    all_activations = {}

    with torch.no_grad():
        for x in tqdm(loader):
            x = x.to(device)
            model(x)

            for k, v in collector.activations:
                all_activations[k] = k if k not in all_activations.keys() \
                    else np.concatenate([all_activations[k], v])

        store_activations(
            all_activations,
            args.out_dir,
            samples_per_image=args.samples_per_image
        )


if __name__ == "__main__":
    main()