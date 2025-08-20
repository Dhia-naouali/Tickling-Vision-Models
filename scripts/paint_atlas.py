import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torchvision.utils import save_image

from src.utils import load_inceptionV1
from src.sae import load_sae
from src.viz import maximize_direction, umap_projection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="latent_data")
    parser.add_argument("--checkpoints-dir", default="SAE_checkpoints")
    parser.add_argument("--atlas-dir", default="atlas")
    parser.add_argument("--directions-per-layer", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_inceptionV1().to(device)

    with open(os.path.join(args.data_dir, "meta.json")) as file:
        meta = json.load(file)

    layer_names = meta["layers"]
    os.makedirs(args.atlas_dir, exist_ok=True)
    layer_name_map = dict(model.named_modules())

    for ln in layer_names:
        layer = layer_name_map[ln]
        sae = load_sae(ln, args.checkpoints_dir, device=device)

        W = sae.decoder.weight.detach().cpu().numpy()
        # directions
        W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-10)

        global_activations = np.load(
            os.path.join(
                args.data_dir,
                f"{ln}_global_activations.npy"
            )
        )

        local_activations = np.load(
            os.path.join(
                args.data_dir,
                f"{ln}_local_activations.npy"
            )
        )

        alignment_scores = (global_activations @ W).var(axis=0)
        topk = np.argsort(-alignment_scores)[:args.directions_per_layer]
        for idx in tqdm(topk):
            direction = torch.tensor([W[:, idx]], dtype=torch.float32, device=device)
            img = maximize_direction(
                model,
                layer, 
                direction, 
                steps=480, 
                lr=0.2,
                device=device,
                l2_lambda=4e-2,
                tv_lambda=4e-2,
            )
            save_image(img, os.path.join(args.atlas_dir, f"{ln}_direction_{idx}.png"))
            u_proj = umap_projection(local_activations, samples_limit=3e3)
            np.save(os.path.join(args.atlas_dir, f"{ln}_direction_{idx}_umap.npy"), u_proj)


if __name__ == "__main__":
    main()