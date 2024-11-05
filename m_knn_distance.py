import torch
from torch.utils.data import DataLoader, TensorDataset

import os
import time
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

def seed_everything(seed=42):
    """Ensure reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_knn(features, k):
    """
    Compute the k-nearest neighbors for the features using pairwise distances.
    Args:
        features (Tensor): Tensor of shape (n_samples, n_features).
        k (int): Number of nearest neighbors to compute.
    Returns:
        Tensor: Indices of the k-nearest neighbors for each sample.
    """
    # Compute pairwise distances using the L2 norm (can be modified to cosine distance)
    distances = torch.cdist(features, features, p=2)  # Shape: (n_samples, n_samples)

    # Get the indices of the k nearest neighbors (excluding the sample itself)
    knn_indices = distances.topk(k + 1, largest=False).indices[:, 1:]  # Shape: (n_samples, k)
    return knn_indices

def mnn_metric(f_features, g_features, k):
    """
    Compute the mutual nearest neighbor (mNN) metric between two feature sets.
    Args:
        f_features (Tensor): Feature tensor from model f (n_samples, n_features).
        g_features (Tensor): Feature tensor from model g (n_samples, n_features).
        k (int): Number of nearest neighbors to use.
    Returns:
        float: The mNN metric score.
    """
    b = f_features.size(0)  # Number of samples in the batch

    # Compute k-nearest neighbors for both feature sets
    knn_f = compute_knn(f_features, k)  # Shape: (n_samples, k)
    knn_g = compute_knn(g_features, k)  # Shape: (n_samples, k)

    # Compute the average intersection of the k-nearest neighbor sets
    total_intersection = 0
    for i in range(b):
        intersection = len(set(knn_f[i].tolist()) & set(knn_g[i].tolist()))
        total_intersection += intersection

    # Compute the average mNN score
    mnn_score = total_intersection / (b * k)
    return mnn_score


def highlight_max_in_rows(data, ax):
    for i in range(data.shape[0]):
        max_idx = np.argmax(data[i, :])
        ax.add_patch(Rectangle((max_idx, i), 1, 1, fill=False, edgecolor='black', linewidth=2))


def plot_similarity_heatmap(df, ouput_path):
    # Compute the mean similarity for each model across all augmentations
    model_ranking = df.groupby('Model')['mean mNN Similarity'].mean().reset_index()
    
    # Sort the models based on the mean similarity (highest to lowest)
    model_ranking = model_ranking.sort_values(by='mean mNN Similarity', ascending=False)
    
    # Extract the ordered models
    ordered_models = model_ranking['Model'].tolist()
    
    # Pivot the dataframe to get augmentations as rows and models as columns
    similarity_matrix = df.pivot(index='Augmentation', columns='Model', values='mean mNN Similarity')
    
    # Reorder the columns based on the model ranking
    similarity_matrix = similarity_matrix[ordered_models]

    # Create a custom colormap
    hex_color1 = "#FFFFFF"  # White
    hex_color2 = "#422c5a"  # Dark purple
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", [hex_color1, hex_color2])

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(similarity_matrix, annot=True, cmap=cmap, cbar_kws={'label': 'Similarity'})

    # Highlight the maximum values in each row
    highlight_max_in_rows(similarity_matrix.to_numpy(), ax)

    # Add title and labels
    plt.title('Control vs Augmentation Similarity Heatmap (Ordered by Mean Similarity)')
    plt.ylabel('Augmentation')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(ouput_path, 'similarity_plot.png'))


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)

    # embd = torch.load(args.embd_path)
    # Load the dictionaries
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

    results = []
    for model_name, aug_dict in embd.items():
        for aug_name, embeddings in aug_dict.items():
            if aug_name != 'rotate_0':
                st = time.time()

                dataset = TensorDataset(embd[model_name]['rotate_0'], embeddings)
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                
                similarities = []
                for index, (xb1, xb2) in enumerate(dataloader):
                    xb1, xb2 = xb1.to(device), xb2.to(device)
                    similarities.append(mnn_metric(xb1, xb2, k=args.k))

                mean_sim = torch.tensor(similarities).mean().item()
                std_sim = torch.tensor(similarities).std().item()
                results.append([model_name, aug_name, mean_sim, std_sim])
                print(model_name, aug_name, time.time() - st)


    df = pd.DataFrame(results, columns=['Model', 'Augmentation', 'mean mNN Similarity', 'std mNN Similarity'])
    # plot_similarity_heatmap(df, os.path.dirname(os.path.abspath(args.embd_path)))

    df.to_csv('m_knn.csv')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View embeddings.')
    parser.add_argument('--embd_path', type=str, help='Path to experiment embeddings.')
    parser.add_argument('--k', type=int, default=50, help='K neighbours for K-mNN metric')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--seed', type=int, default=0, help='Seed value for random.')
    args = parser.parse_args()
 
    main(args)