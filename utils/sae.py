from torch import nn
import torch.nn.functional as F

class SAE(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.encoder = nn.Linear(in_dim, z_dim)
        self.decoder = nn.Linear(in_dim, z_dim)
        self.apply(self.init_weights)

    def init_weights(self, module):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)

    def forward(self, x):
        z = self.encoder(x)
        z = F.relu(z)
        return self.decoder(z), z

