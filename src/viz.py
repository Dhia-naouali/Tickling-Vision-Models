import umap
import torch
import numpy as np


def tv(x):
    dx = (x[...,1:,:]-x[...,:-1,:]).abs().mean()
    dy = (x[...,:,1:]-x[...,:, :-1]).abs().mean()
    return dx+dy

def tv_(x):
    dx = (x[...,1:,:]-x[...,:-1,:]).pow(2)
    dy = (x[...,:,1:]-x[...,:, :-1]).pow(2)
    return (dx + dy).sqrt().mean()


def maximize_direction(
    model, 
    layer, 
    direction,
    lr=2e-8, 
    steps=320,
    device="cpu", 
    img_size=224,
    l2_lambda=1e-3,
    tv_lambda=1e-3,
):
    means = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, -1, 1, 1)
    stds  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, -1, 1, 1)
    
    collector = {}
    def hook(module, input_, output):
        global_repr = output.mean(dim=(2, 3)).squeeze(0) # hidden_dim
        stimulation_score = (global_repr * direction).sum()
        collector["ss"] = stimulation_score

    handle = layer.register_forward_hook(hook)
    x = torch.randn(1, 3, img_size, img_size, requires_grad=True).to(device) * 0.1 + 0.5
    x.retain_grad()
    optimizer = optim.Adam([x], lr)
    for _ in range(steps):
        collector.clear()
        optimizer.zero_grad()
        x_c = torch.clamp(x, 0, 1)
        x_n = (x_c - means) / (stds + 1e-8)
        model(x_n)
        loss = -collector["ss"] + l2_lambda * (x.pow(2)).mean() + tv_lambda * tv(x)
        loss.backward(retain_graph=True)
        # print(x.grad.shape)
        optimizer.step()
    handle.remove()
    return torch.sigmoid(x.detach()).cpu()


def umap_projection(local_A, samples_limit=2e3):
    if len(local_A) > samples_limit:
        idx = np.random.choice(len(local_A), int(samples_limit), replace=False)
        local_A = local_A[idx]
    reducer = umap.UMAP(32, metric="cosine")
    return reducer.fit_transform(local_A)