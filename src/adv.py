import torch
import torch.nn.functional as F


def fgsm(model, x, y, alpha=1e-2):
    model.zero_grad(set_to_none=True)
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    F.cross_entropy(logits, y).backward()
    x_adv = x + alpha * x.grad.sign()
    return x_adv.clamp(0, 1).detach()


def pgd(model, x, y, steps=10, eps=1e-2, alpha=1e-3):
    x_adv = (x.clone() + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(x_adv)
        F.cross_entropy(logits, y).backward()

        with torch.no_grad():
            x_adv += alpha * x_adv.grad.sign()
            x_adv = clamp_in_ball(x, x_adv, eps)
            x_adv = x_adv.clamp(0, 1)
    return x_adv.detach()


def clamp_in_ball(origin, adv, eps):
    return  torch.max(
                origin - eps,
                torch.min(adv, origin + eps)
            )