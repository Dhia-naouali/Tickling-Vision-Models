import os
import json
import argparse

import torch

from src.utils import load_inceptionV1, imagenet_preprocess, setup_loader
from src.patching import ablate_directions
from src.adv import pgd, fgsm
from src.sae import load_sae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default="inception4a")
    parser.add_argument("--raw-data-dir", default="__download__")
    parser.add_argument("--adv-dir", default="advs")
    parser.add_argument("--checkpoints-dir", default="SAE_checkpoints")
    parser.add_argument("--pgd-steps", type=int, default=12)
    parser.add_argument("--ablate-p", type=float, default=0.25)    
    parser.add_argument("--num-samples", type=int, default=64)    
    args = parser.parse_args()

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_inceptionV1(device)
    preprocess = imagenet_preprocess()
    layer = dict(model.named_modules())[args.layer]
    
    sae = load_sae(args.layer, args.checkpoints_dir, device=device)
    D = sae.decoder.weight.detach()
    ablate_k = int(D.shape[1] * args.ablate_p)

    loader = setup_loader(args.raw_data_dir, preprocess, batch_size=1)    
    original_activations = {}
    adv_activations = {}
    def original_hook(model, input_, output):
        original_activations["x"] = output.detach()

    def adv_hook(model, input_, output):
        adv_activations["x"] = output.detach()

    records = []
    for x, y in loader:
        args.num_samples -= 1
        if not args.num_samples: break
        
        x = x.to(device).requires_grad_(True)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            

        x_adv = pgd(model, x.detach(), pred, steps=args.pgd_steps) if args.pgd_steps > 1 \
            else fgsm(model, x.detach(), pred)
            
        
        original_handle = layer.register_forward_hook(original_hook)
        model(x); original_handle.remove()
        
        adv_handle = layer.register_forward_hook(adv_hook)
        adv_logits = model(x_adv); adv_handle.remove()
        
        A = original_activations["x"]
        adv_A = adv_activations["x"]
        
        c = adv_A.shape[1]
        A = A.permute(0, 2, 3, 1).reshape(-1, c)
        adv_A = adv_A.permute(0, 2, 3, 1).reshape(-1, c)
        delta = adv_A - A
        
        sparse_delta = torch.matmul(delta, D)
        magnitude = sparse_delta.abs().mean(dim=0)
        topk_idx = torch.topk(magnitude, k=ablate_k).indices.cpu().tolist()
        
        D_top = D[:, topk_idx]
        
        def ablate_hook(module, input_, output):
            return ablate_directions(output, D_top)
        
        ablation_handle = layer.register_forward_hook(ablate_hook)
        ablation_logits = model(x_adv); ablation_handle.remove()
        
        records.append({
            "ground_truth": y.item(),
            "original_pred": pred.item(),
            "adv_pred": adv_logits.argmax(dim=1).item(),
            "abl_adv_pred": ablation_logits.argmax(dim=1).item(),
            "top_blank": topk_idx,
            "delta_abl_origin_logits": ablation_logits.max().item() - logits.max().item(),
            "delta_abl_adv_logits": ablation_logits.max().item() - adv_logits.max().item()
        })

    os.makedirs(args.adv_dir, exist_ok=True)
    with open(os.path.join(args.adv_dir, f"{args.layer}_adversarial.json"), "w") as file:
        json.dump(records, file, indent=2)


if __name__ == "__main__":
    main()