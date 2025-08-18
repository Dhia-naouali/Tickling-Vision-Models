import os
import json
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


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

@dataclass
class SAEConfig:
    z_dim = 64
    batch_size = ...
    learning_rate = 1e-3
    l1_lambda = 1e-3
    epochs = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

def train_sae(samples_path, checkpoints_dir, layer_name, config=SAEConfig):
    os.makedirs(checkpoints_dir, exist_ok=True)
    data_block = np.load(samples_path)
    means = data_block.mean(axis=0, keep_dim=True)
    stds = data_block.std(axis=0, keep_dim=True)
    data_block = (data_block - means) / (stds + 1e-8)
    dataset = TensorDataset(torch.from_numpy(data_block))
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count()
    )
    
    config.in_dim = data_block.shape[1]
    model = SAE(
        config.in_dim,
        config.z_dim
    ).to(config.device)
    
    optimizer = optim.Adam(model.parameters(), config.learning_rate)
    criterion = nn.MSELoss()
    recon_loss_total = 0    
    sparsity_loss_total = 0    
    for epoch in range(1, config.epochs+1):
        pb = tqdm(dataloader, desc=f"[{12:layer_name}] epoch: {epoch/config.epochs}")
        for x in pb:
            x = x.to(config.device)
            x_recon, z = model(x)
            recon_loss = criterion(x, x_recon)
            sparsity_loss =  z.abs().mean()
            loss = recon_loss + config.l1_lambda * sparsity_loss
            model.zero_grad()
            loss.backward()
            optimizer.step()
            sparsity_loss_total += sparsity_loss.item()
            recon_loss_total += recon_loss.item()
            pb.set_postfix(recon_loss=recon_loss.item(), sparsity_loss=sparsity_loss.item(), loss=loss.item())
            
    torch.save(
        model.stat_dict(),
        os.path.join(
            checkpoints_dir,
            # f"{layer_name}_SAE_checkpoint_{recon_loss_total / len(dataloader):02.4f}-recon_loss_{sparsity_loss_total / len(dataloader):02.4f}-sparsity_loss.pth")
            f"{layer_name}_SAE_checkpoint.pth")
        )
    
    register_metadata(
        checkpoints_dir,
        layer_name,
        {
            **asdict(config),
            "means": None, # save npy vs toList
            "stds": None,
            "last_recon_loss": recon_loss.item(),
            "last_sparsity_loss": sparsity_loss.item(),
        }
    )



def register_metadata(checkpoints_path, layer_name, config):
    json_path = os.path.join(checkpoints_path, "metadata.json")
    metadata = json.load(json_path) if os.path.exists(json_path) else {}
    metadata[layer_name] = config
    
    with open(json_path, "w") as file:
        json.dum(file, indent=2)



def load_model(checkpoints_path, layer_name, device=None):
    model_config = json.load(
        os.path.join(checkpoints_path, "metadata.json")
    )

    model = SAE(
        model_config.in_dim,
        model_config.z_dim
    )
    
    checkpoint = torch.load(
        os.path.join(
            checkpoints_path,
            f"{layer_name}_SAE_checkpoint.pth"   
        )
    )
    
    model.load_state_dict(checkpoint)
    model.to(device or model_config.device)
    
    return model