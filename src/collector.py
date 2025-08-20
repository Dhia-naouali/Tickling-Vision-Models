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
        modules = dict(self.model.named_modules())
        for layer_name in self.layer_names:
            module = modules[layer_name]
            self.handles.append(
                module.register_forward_hook(
                    self.attach_hook(layer_name)
                )
            )

    def attach_hook(self, name):
        def hook(module, input_, output):
            self.activations[name] = output.detach()
        return hook

    def clear(self):
        self.activations = {}

    def detach_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def sample_pixels(a, samples_per_image=512):
    *_, h, w = a.shape
    n = min(samples_per_image, h*w)
    w_idx = torch.randint(0, w, (n,))
    h_idx = torch.randint(0, h, (n,))
    return a[:, :, h_idx, w_idx]


def store_activations(activations, labels, out_dir, samples_per_image=512):
    os.makedirs(out_dir, exist_ok=True)
    for ln, a in activations.items():
        batched_samples = sample_pixels(a, samples_per_image=samples_per_image)
        samples = batched_samples.transpose(0, 2, 1).reshape(-1, batched_samples.shape[1])
        np.save(
            os.path.join(out_dir, f"{ln}_local_activations.npy"), 
            samples.cpu().numpy() if isinstance(samples, torch.Tensor) else samples
        )

        samples = batched_samples.mean(axis=2)
        np.save(
            os.path.join(out_dir, f"{ln}_global_activations.npy"), 
            samples.cpu().numpy() if isinstance(samples, torch.Tensor) else samples
        )

        np.save(
            os.path.join(out_dir, f"labels.npy"), 
            labels
        )

    meta = {"layers": list(activations.keys()), "samples_per_image": samples_per_image}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)



def checkpoint_activations(gpu_A, cpu_A):
    for k, v in gpu_A.items():
        v = v.cpu().numpy()
        if k in cpu_A:
            cpu_A[k] = np.concatenate([cpu_A[k], v])
        else:
            cpu_A[k] = v
    gpu_A.clear()
    torch.cuda.empty_cache()