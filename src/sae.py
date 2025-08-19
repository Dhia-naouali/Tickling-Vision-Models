import os
import json
import numpy as np
from tqdm.auto import tqdm
from typing import Union
from dataclasses import dataclass, asdict

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class SAE(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.encoder = nn.Linear(in_dim, z_dim)
        self.decoder = nn.Linear(z_dim, in_dim)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        z = self.encoder(x)
        z = F.relu(z)
        return self.decoder(z), z

@dataclass
class SAEConfig:
    z_dim: int = 64
    batch_size: int = 32
    learning_rate: float = 1e-3
    l1_lambda: float = 1e-3
    epochs: int = 4
    device: Union[torch.device, str] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

def train_sae(samples_path, checkpoints_dir, layer_name, config=SAEConfig):
    os.makedirs(checkpoints_dir, exist_ok=True)
    data_block = np.load(samples_path)
    means = data_block.mean(axis=0, keepdims=True)
    stds = data_block.std(axis=0, keepdims=True)
    data_block = (data_block - means) / (stds + 1e-8)
    dataset = TensorDataset(torch.from_numpy(data_block))
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    
    model = SAE(
        data_block.shape[1], 
        config.z_dim
    ).to(config.device)
    
    optimizer = optim.Adam(model.parameters(), config.learning_rate)
    criterion = nn.MSELoss()
    recon_loss_total = 0    
    sparsity_loss_total = 0    
    for epoch in range(1, config.epochs+1):
        pb = tqdm(dataloader, desc=f"[{layer_name:<12}] epoch: {epoch}/{config.epochs}", ncols=160)
        for (x,) in pb:
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
        model.state_dict(),
        os.path.join(
            checkpoints_dir,
            f"{layer_name}_SAE_checkpoint_{recon_loss_total / len(dataloader):02.4f}-recon_loss_{sparsity_loss_total / len(dataloader):02.4f}-sparsity_loss.pth")
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
    metadata = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as file:
            metadata = json.load(file)
    
    
    metadata[layer_name] = {**config, "device": config["device"].type}
    with open(json_path, "w") as file:
        json.dump(metadata, file, indent=2)

def load_model(layer_name):
    ...