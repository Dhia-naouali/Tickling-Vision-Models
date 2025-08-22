import torch

def swap_direction(target_A, donor_A, direction):
    direction = direction.view(1,-1,1,1).to(target_A.device)
    coef_d = (donor_A * direction).sum(dim=1, keepdim=True)
    coef_t = (target_A * direction).sum(dim=1, keepdim=True)
    return target_A + (coef_d - coef_t) * direction


def ablate_directions(A, adv_A, D):
    D /= (D.norm(dim=0, keepdim=True) + 1e-8)
    
    W = torch.matmul(A, D)
    adv_W = torch.matmul(adv_A, D)
    
    V = torch.matmul(W, D.T)
    adv_V = torch.matmul(adv_W, D.T)

    return adv_A - adv_V + V