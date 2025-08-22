import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.metrics import confusion_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-dir", default="atlas")
    
    args = parser.parse_args()
    
    
    with open(glob.glob(os.path.join(args.atlas_dir, "*_patching_results.json"))[0]) as file:
        patching_records = json.load(file)

    df = pd.DataFrame(patching_records)

    pairs = [(row["donor_target"], row["target_label"], int(row["donor_target"] in row["top5_classes"]))
            for row in patching_records]

    matrix = defaultdict(lambda: defaultdict(int))
    counts = defaultdict(lambda: defaultdict(int))

    for d, t, x in pairs:
        matrix[d][t] += x
        counts[d][t] += 1


    donors  = sorted({d for d, _, _ in pairs})
    targets = sorted({t for _, t, _ in pairs})
    rates_matrix   = np.zeros((len(donors), len(targets)))


    for i, d in enumerate(donors):
        for j, t in enumerate(targets):
            rates_matrix[i, j] = matrix[d][t]


    plt.figure(figsize=(10, 8))
    sns.heatmap(
        rates_matrix,
        xticklabels=targets,
        yticklabels=donors,
        annot=True,
        fmt=".2f",
    )
    plt.title("Donor-Target influence based on top5 logits")
    plt.ylabel("Donor class")
    plt.xlabel("Target class")
    plt.savefig(os.path.join(args.atlas_dir, "donor_influence_matrix.png"))

    shifts = []
    for r in patching_records:
        if r["donor_target"] in r["top5_classes"]:
            idx = r["top5_classes"].index(r["donor_target"])
            shifts.append(r["top5_probs"][idx])

    plt.hist(shifts, bins=20, color="blue", alpha=0.6)
    plt.title("Distribution of donor probability in top5 logits")
    plt.xlabel("probability")
    plt.ylabel("frequency")
    plt.savefig(os.path.join(args.atlas_dir, "donor_prob_dist.png"))



    with open(glob.glob(os.path.join(args.advs_dir, "*_adversarial.json"))[0], "r") as f:
        adv_records = json.load(f)

    ground_truth = np.array([r["ground_truth"] for r in adv_records])
    adv_pred = np.array([r["adv_pred"] for r in adv_records])
    abl_adv_pred = np.array([r["abl_adv_pred"] for r in adv_records])
    delta_abl_origin = np.array([r["delta_abl_origin_logits"] for r in adv_records])
    delta_abl_adv = np.array([r["delta_abl_adv_logits"] for r in adv_records])
    top_hijjacked = [tuple(r["top_hijjacked"]) for r in adv_records]

    top_hijjacked = np.array(top_hijjacked).flatten().tolist()
    counts = Counter(top_hijjacked)
    dirs, freq = zip(*counts.most_common())
    plt.figure(figsize=(10,4))
    sns.barplot(x=list(dirs), y=list(freq), palette="rocket")
    plt.title("Frequency of top hijacked directions")
    plt.xlabel("Direction index")
    plt.ylabel("number of samples")
    plt.savefig(os.path.join(args.atlas_dir, "directions_hijack_freq.png"))


    plt.figure(figsize=(6,6))
    plt.scatter(delta_abl_origin, delta_abl_adv, alpha=0.7, color="blue")
    plt.xlabel("Delta ablation vs original logits")
    plt.ylabel("Delta ablation vs adversarial logits")
    plt.title("Ablation effectiveness per sample")
    plt.axhline(0, color='red', linestyle='--', alpha=.5)
    plt.axvline(0, color='red', linestyle='--', alpha=.5)
    plt.savefig(os.path.join(args.atlas_dir, "ablation_effectiveness.png"))


    recovered_idx = np.where((adv_pred != ground_truth) & (abl_adv_pred == ground_truth))[0]
    if len(recovered_idx) > 0:
        cm = confusion_matrix(ground_truth[recovered_idx], abl_adv_pred[recovered_idx])
        plt.title("confusion matrix of recovered samples")
        sns.heatmap(cm, annot=True, cmap="bone")
        plt.savefig(os.path.join(args.atlas_dir, "ablation_recovery_confusion_matrix.png"))
        
        
        
if __name__ == "__main__":
    main()