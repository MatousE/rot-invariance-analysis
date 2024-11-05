import torch
import torch.nn.functional as F

import argparse
import numpy as np
import pandas as pd

from utils.structure_analysis import *

def cosine_distance(tensor1, tensor2):
    # Normalize each vector along the feature dimension
    tensor1_norm = F.normalize(tensor1, p=2, dim=1)
    tensor2_norm = F.normalize(tensor2, p=2, dim=1)
    
    # Compute cosine similarity
    cosine_sim = torch.sum(tensor1_norm * tensor2_norm, dim=1)
    
    # Convert cosine similarity to cosine distance
    cosine_dist = 1 - cosine_sim
    
    return cosine_dist

def seed_everything(seed=42):
    """Ensure reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(args):
    seed_everything(args.seed)
    embd_1 = torch.load('results/embeddings/final_embeddings_20241101.pth')
    embd_2 = torch.load('results/embeddings/final_embeddings_20241103.pth')

    # Initialize the merged dictionary with dict1
    embd = embd_1.copy()

    # Merge dict2 into merged_dict
    for model_key, augmentations in embd_2.items():
        if model_key not in embd:
            embd[model_key] = augmentations
        else:
            for aug_key, value in augmentations.items():
                embd[model_key][aug_key] = value

    # embd = torch.load(args.embd_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for model_name in embd.keys():  # Outer directory is now "models"
        for aug_name in embd[model_name].keys():  # Inner directory is "augmentations"
            if aug_name != 'rotate_0':
                ctrl_sample = embd[model_name]['rotate_0'].to(device)
                aug_sample = embd[model_name][aug_name].to(device)

                # cos_dist = torch.mean(1 - F.cosine_similarity(ctrl_sample, aug_sample))
                cos_dist = torch.mean(cosine_distance(ctrl_sample, aug_sample))
                results.append([model_name, aug_name, cos_dist.item()])

    df = pd.DataFrame(results, columns=['Model', 'Augmentation', 'cosine distance'])

    df.to_csv('cosine.csv')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View embeddings.')
    parser.add_argument('--embd_path', type=str, help='Path to experiment embeddings.')
    parser.add_argument('--seed', type=int, default=0, help='Seed value for random.')
    parser.add_argument('--save_dir', type=str, default='results/', help='Output dir.')

    args = parser.parse_args()
    main(args)
