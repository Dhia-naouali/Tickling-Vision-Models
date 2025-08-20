import torch

def swap_direction(target_A, donor_A, direction):
    direction = direction.view(1,-1,1,1).to(target_A.device)
    coef_d = (donor_A * direction).sum(dim=1, keepdim=True)
    coef_t = (target_A * direction).sum(dim=1, keepdim=True)
    return target_A + (coef_d - coef_t) * direction