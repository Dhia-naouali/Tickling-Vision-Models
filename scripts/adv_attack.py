import os
import json
import argparse

import torch

from src.utils import load_inceptionV1, imagenet_preprocess, setup_loader
from src.adv import pgd, fgsm
from src.sae import load_sae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default="inception4a")
    parser.add_argument("--raw-data-dir", default="__download__")
    parser.add_argument("--adv-dir", default="advs")
    parser.add_argument("--checkpoints-dir", default="SAE_checkpoints")
    parser.add_argument("--data-dir")
    parser.add_argument("--pgd-steps", type=int, default=12)
    parser.add_argument("--ablate-k", type=int, default=12)
    
    
    args = parser.parse_args()
    
    num_images = 12
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_inceptionV1(device)
    preprocess = imagenet_preprocess()
    layer_name_map = dict(model.named_modules())
    layer = layer_name_map[args.layer]
    
    sae = load_sae(args.layer, args.checkpoints_dir, device=device)
    D = sae.decoder.weight.deatch()
    
    dataset = setup_loader(args.raw_data_dir, preprocess).dataset[:num_images]
    def loader(dataset):
        for i in range(len(dataset)):
            yield preprocess(dataset[i]["image"]), dataset[i]["label"]
    
    records = []
    for x, _ in loader(dataset):
        x.requires_grad_(True)
        with torch.no_grad():
            logits = model(x)
            y = logits.argmax(dim=1)
            
        if args.pgd_steps - 1:
            x_adv = pgd(model, x.detach(), y, steps=args.pgd_steps)
        else:
            x_adv = fgsm(model, x.detach(), y)
            
        original_activations = {}
        adv_activations = {}
        def original_hook(model, input_, output):
            original_activations["x"] = output.detach()

        def adv_hook(model, input_, output):
            adv_activations["x"] = output.detach()
        
        original_handle = layer.register_forward_hook(original_hook)
        model(x)
        original_handle.remove()
        
        adv_handle = layer.register_forward_hook(adv_hook)
        adv_logits = model(x_adv)
        adv_handle.remove()
        
        a = original_activations["x"]
        b = adv_activations["x"]
        
        bs, c, h, w = b.shape
        a = a.permute(0, 2, 3, 1).reshape(-1, c)
        b = b.permute(0, 2, 3, 1).reshape(-1, c)
        delta = b - a
        
        sparse_delta = torch.matmul(delta, D.t())
        energy = sparse_delta.abs().mean(dim=0)
        topk = torch.topk(energy, k=min(args.ablate_k, energy.numel())
        
        topk_idx = topk.indices.cpu().numpy().tolist()
        
        D_top = D[topk_idx]
        
        def ablate_hook(module, input_, output):
            return ablate_directions(output, D_top, topk=None)
        
        ablation_handle = layer.register_forward_hook(ablate_hook)
        ablation_logits = model(x_adv)
        ablation_handle.remove()
        
        records.append({
            "original_pred": y.item(),
            "adv_pred": adv_logits.argmax(dim=1),
            "abl_adv_pred": ablation_logits.argmax(dim=1),
            "top_blank": topk_idx,
            "delta_abl_origin_logits": ablation_logits.max().item() - logits.max().item(),
            "delta_abl_adv_logits": ablation_logits.max().item() - adv_logits.max().item()
        })

    os.makedirs(args.adv_dir, exist_ok=True)
    with open(os.path.join(args.adv_dir), f"{args.layer}_adversarial.json", "w") as file:
        json.dump(reconds, file, indent=2)


if __name__ == "__main__":
    main()