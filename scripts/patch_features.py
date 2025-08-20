import os
import json
import tqdm
import random
import argparse
import numpy as np

import torch

from src.sae import load_sae
from src.patching import swap_direction
from src.utils import load_inceptionV1, imagenet_preprocess, setup_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default="inception4a")
    parser.add_argument("--data-dir", default="__download__")
    parser.add_argument("--atlas-dir", default="atlas")
    parser.add_argument("--num-pairs", type=int, default=12)

    ...
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_inceptionV1(device)
    preprocess = imagenet_preprocess()
    dataset = setup_loader(args.data_dir, preprocess).dataset
    get_sample = lambda i: preprocess(dataset[i]["image"]), dataset[i]["label"]
    indices = list(range(
        min(
            len(dataset) - len(dataset) % 2,
            2*args.num_pairs)
    ))
    random.shuffle(indices)
    p1 = indices[::2]
    p2 = indices[1::2]
    pairs = [
        (get_sample(p1), get_sample(p2)) for p1, p2 in zip(p1, p2)
    ]
    
    if not f"{args.layer}_global_activations.npy" in os.listdir(args.data_dir):
        raise ValueError(f"invalid layer name, no activations found in {args.data_dir}")
    global_activations = np.load(os.path.join(args.data_dir, f"{args.layer}_global_activations.npy"))
    sae = load_sae(args.layer)

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
    layer_ = layer_name_map(args.layer)
    results = []
    with torch.no_grad():
        for (donor, dlabel), (target, tlabel) in tqdm(pairs):
            donor = preprocess(donor).to(device)
            target = preprocess(target).to(device)
            
            donor_handle = layer_.register_forward_hook(donor_hook)    
            model(donor)
            donor_handle.remove()
            
            target_handle = layer_.register_forward_hook(target_hook)
            logits = model(target)
            target_handle.remove()
            
            probs = logits.softmax(dim=1)
            top5 = torch.topk(probs, k=5)
            results.append({
                "donor_target": dlabel,
                "target_label": tlabel,
                "direction_index": direction_idx,
                "top5_classes": top5.indices.tolist(),
                "top5_probs": top5.values.tolist()
            })
            
    os.makedirs(args.atlas_dir)
    with open(os.path.join(args.atlas_dir, f"{args.layer}_patching_results.json"), "w") as file:
        json.dump(results, file, indent=2)
    

if __name__ == "__main__":
    main()