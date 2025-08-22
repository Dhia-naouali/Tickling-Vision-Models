import os
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch

from src.sae import load_sae
from src.patching import swap_direction
from src.utils import load_inceptionV1, imagenet_preprocess, setup_loader, CLASSES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default="inception4a")
    parser.add_argument("--raw-data-dir", default="__download__")
    parser.add_argument("--checkpoints-dir", default="SAE_checkpoints")
    parser.add_argument("--data-dir", default="latent_data")
    parser.add_argument("--atlas-dir", default="atlas")
    parser.add_argument("--num-pairs", type=int, default=512)

    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_inceptionV1(device)
    preprocess = imagenet_preprocess()
    dataset = setup_loader(args.raw_data_dir, preprocess).dataset
    def get_sample(i):
        return preprocess(dataset[i]["image"].convert("RGB")), dataset[i]["label"]
        

    indices = random.sample(
        list(range(len(dataset))),
        min(
            len(dataset) - len(dataset) % 2,
            2*args.num_pairs
        )
    )

    p1 = indices[::2]
    p2 = indices[1::2]
    pairs = [
        (get_sample(p1), get_sample(p2)) for p1, p2 in zip(p1, p2)
    ]
    
    if not f"{args.layer}_global_activations.npy" in os.listdir(args.data_dir):
        raise ValueError(f"invalid layer name, no activations found in {args.data_dir}")
    global_activations = np.load(os.path.join(args.data_dir, f"{args.layer}_global_activations.npy"))
    sae = load_sae(args.layer, args.checkpoints_dir)

    D = sae.decoder.weight.detach().cpu().numpy()
    D /= np.linalg.norm(D, axis=0, keepdims=True) + 1e-8
    stim_scores = (global_activations @ D).var(axis=0)
    direction_idx = np.argmax(stim_scores)
    direction = torch.tensor(D[:, direction_idx], dtype=torch.float32, device=device)
    
    collector = {}
    def donor_hook(module, input_, output):
        collector["donor"] = output.detach()

    def target_hook(module, input_, output):
        return swap_direction(output, collector["donor"], direction)
        
    layer_name_map = dict(model.named_modules())
    layer_ = layer_name_map[args.layer]
    results = []
    with torch.no_grad():
        for (donor, dlabel), (target, tlabel) in tqdm(pairs):
            donor = donor.to(device).unsqueeze(0)
            target = target.to(device).unsqueeze(0)
            
            donor_handle = layer_.register_forward_hook(donor_hook)    
            model(donor)
            donor_handle.remove()
            
            target_handle = layer_.register_forward_hook(target_hook)
            logits = model(target)
            target_handle.remove()
            
            logits = logits[0, CLASSES]
            probs = logits.softmax(dim=0)
            top5 = torch.topk(probs, k=5)
            top5_indices = [CLASSES[i] for i in top5.indices.tolist()]
            results.append({
                "donor_target": dlabel,
                "target_label": tlabel,
                "direction_index": direction_idx.item(),
                "top5_classes": top5_indices,
                "top5_probs": top5.values.tolist()
            })
            
    os.makedirs(args.atlas_dir, exist_ok=True)
    with open(os.path.join(args.atlas_dir, f"{args.layer}_patching_results.json"), "w") as file:
        json.dump(results, file, indent=2)
    

if __name__ == "__main__":
    main()