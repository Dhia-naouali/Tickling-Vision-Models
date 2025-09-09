#!/bin/bash
set -e

python -m scripts.collect_activations
python -m scripts.train_decomposers
python -m scripts.paint_atlas
python -m scripts.patch_features
python -m scripts.adversarial_attack_ablation