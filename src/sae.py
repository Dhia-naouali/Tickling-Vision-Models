import os
import json
import glob
import numpy as np
from tqdm import tqdm
from typing import Union
from dataclasses import dataclass, asdict

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class SAE(nn.Module):
    def __init__(self, in_dim, z_dim_factor):
        super().__init__()
        z_dim = in_dim // z_dim_factor
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
    in_dim: int = None
    z_dim:  int = None
    z_dim_factor: int = 32
    batch_size: int = 64
    learning_rate: float = 3e-4
    l1_lambda: float = 5e-1
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
        config.z_dim_factor
    ).to(config.device)
    config.in_dim = data_block.shape[1]
    config.z_dim = config.in_dim / config.z_dim_factor
    optimizer = optim.Adam(model.parameters(), config.learning_rate)
    criterion = nn.MSELoss()
    recon_loss_total = 0    
    sparsity_loss_total = 0    
    for epoch in range(1, config.epochs+1):
        pb = tqdm(dataloader, desc=f"[{layer_name} {epoch}/{config.epochs}]", ncols=160)
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
    print("="*160)

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


def load_sae(layer_name, checkpoints_dir, device="cpu"):
    with open(os.path.join(checkpoints_dir, f"metadata.json")) as f:
        meta = json.load(f)[layer_name]
    in_dim, z_dim_factor = meta["in_dim"], meta["z_dim_factor"]
    model = SAE(in_dim, z_dim_factor)
    checkpoint_path = glob.glob(os.path.join(checkpoints_dir, f"{layer_name}_SAE*.pth"))[0]
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    return model.to(device).eval()