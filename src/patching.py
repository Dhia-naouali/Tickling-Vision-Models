import torch

def swap_direction(target_A, donor_A, direction):
    direction = direction.view(1,-1,1,1).to(target_A.device)
    coef_d = (donor_A * direction).sum(dim=1, keepdim=True)
    coef_t = (target_A * direction).sum(dim=1, keepdim=True)
    return target_A + (coef_d - coef_t) * direction * 10


def ablate_directions(A, D):
    b, c, h, w = A.shape
    A = A.permute(0, 2, 3, 1).view(-1, c)

    D /= (D.norm(dim=0, keepdim=True) + 1e-8)
    W = torch.matmul(A, D) 
    V = torch.matmul(W, D.T)

    return (A - V).view(b, h, w, c).permute(0, 3, 1, 2)