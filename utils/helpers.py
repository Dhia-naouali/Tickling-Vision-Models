import torch
import numpy as np

class Collector:
    def __init__(self, module):
        self.module = module
        self.activation = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input_, output):
        self.activations = output.detach().cpu()

    def clear(self):
        self.activations = None

    def remove(self):
        self.hook.remove()

def collect_activations(model, loader, layer_module=None, layer_name=None, num_batches=None):
    if not ((layer_module is None) ^ (layer_name is None)):
        raise Exception("should pass layer_module xor layer_name")

    if layer_name:
        layer_module = dict(model.named_modules())[layer_name]

    model.eval()
    device = next(model.parameters()).device
    farmer = Collector(layer_module)
    activations = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            _ = model(imgs.to(device))
            As = farmer.activations
            b, c, = As.shape[:2]
            A = A.view(b, c, -1).mean(dim=2)
            activations.append(As.numpy())
            farmer.clear()
            if num_batches is not None and i == num_batches:
                break
    farmer.remove()
    return np.stack(activations)