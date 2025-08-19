import os
import glob
import argparse
from dataclasses import fields

from src.sae import SAEConfig, train_sae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="SAE_checkpoints")
    parser.add_argument("--z-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--l1-lambda", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    args_d = vars(args)
    args_d["device"] = None
    config_kwargs = {
        k.name: args_d[k.name] for k in fields(SAEConfig)
    }
    
    config = SAEConfig(
        **config_kwargs
    )
    
    for np_block_path in glob.glob(os.path.join("data_dir", "*_activations.npy")):
        layer_name = os.path.base_name(np_block_path).replace("_activations.npy", "")
        train_sae(np_block_path, args.checkpoint_dir, layer_name, config=config)
        
        
if __name__ == "__main__":
    main()