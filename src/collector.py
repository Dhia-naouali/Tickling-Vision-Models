import os
import json
import torch
import numpy as np


class ActivationsCollector:
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names
        self.activations = {}
        self.handles = []
        self._set_hooks()


    def _set_hooks(self):
        modules = self.model
        for layer_name in self.layer_names:
            module = modules[layer_name]
            self.handles.append(
                module.regitser_forward_hook(
                    self.attach_hook(layer_name)
                )
            )

    def attach_hook(self, name):
        def hook(module, input_, output):
            self.activations[name] = output.detach().cpu().numpy()
        return hook

    def clear(self):
        self.activations = {}

    def detach_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def sample_pixels(a, samples_per_image=512):
    b, _, h, w = a.shape
    n = min(samples_per_image, h*w) * b
    b_idx = torch.randint(0, b, (n,))
    w_idx = torch.randint(0, w, (n,))
    h_idx = torch.randint(0, h, (n,))
    return a[b_idx, :, h_idx, w_idx]


def store_activations(activations, out_dir, samples_per_image=512):
    os.makedirs(out_dir, exist_ok=True)
    for ln, a in activations.items():
        if a.dim() == 3: 
            a = a.unsqueeze(0)

        samples = sample_pixels(a, sampes_per_image=samples_per_image)
        np.save(
            os.path.join(out_dir, f"{ln}_activations.npy"), 
            samples.cpu().numpy()
        )

    meta = {"layers": list(activations.keys()), "samples_per_image": samples_per_image}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)














# def collect_activations(model, loader, layer_module=None, layer_name=None, num_batches=None):
#     if not ((layer_module is None) ^ (layer_name is None)):
#         raise Exception("should pass layer_module xor layer_name")

#     if layer_name:
#         layer_module = dict(model.named_modules())[layer_name]

#     model.eval()
#     device = next(model.parameters()).device
#     farmer = ActivationsCollector(layer_module)
#     activations = []
#     with torch.no_grad():
#         for i, (imgs, _) in enumerate(loader):
#             _ = model(imgs.to(device))
#             As = farmer.activations
#             b, c, = As.shape[:2]
#             A = A.view(b, c, -1).mean(dim=2)
#             activations.append(As.numpy())
#             farmer.clear()
#             if num_batches is not None and i == num_batches:
#                 break
#     farmer.remove()
#     return np.stack(activations)