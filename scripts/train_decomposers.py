import os
import glob
import argparse
from dataclasses import fields

from src.sae import SAEConfig, train_sae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="SAE_checkpoints")
    parser.add_argument("--data-dir", default="latent_data")
    parser.add_argument("--z-dim-factor", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--l1-lambda", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    args_d = vars(args)
    args_d["device"] = None
    config_fields = [f.name for f in fields(SAEConfig)]

    config_kwargs = {
        k: args_d[k]  
        for k in config_fields if args_d[k] is not None
    }

    config = SAEConfig(
        **config_kwargs
    )
    
    for np_block_path in glob.glob(os.path.join(args.data_dir, "*_local_activations.npy")):
        layer_name = os.path.basename(np_block_path).replace("_local_activations.npy", "")
        train_sae(np_block_path, args.checkpoints_dir, layer_name, config=config)
        
        
if __name__ == "__main__":
    main()