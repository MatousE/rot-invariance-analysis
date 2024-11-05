import ot
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ot import wasserstein_1d
from scipy.stats import pearsonr



def distance_stats(pre, post, downsample=False, verbose=True):
    """
    Tests for correlation between Euclidean cell-cell distances before and after 
    transformation by a function or DR algorithm.

    Parameters
    ----------

    pre : np.array
        vector of unique distances (pdist()) or distance matrix of shape (n_cells, 
        m_cells), i.e. (cdist()) before transformation/projection
    post : np.array
        vector of unique distances (pdist()) or distance matrix of shape (n_cells, 
        m_cells), i.e. (cdist()) after transformation/projection
    downsample : int, optional (default=False)
        number of distances to downsample to. maximum of 50M (~10k cells, if 
        symmetrical) is recommended for performance.
    verbose : bool, optional (default=True)
        print progress statements to console

    Returns
    -------

    pre : np.array
        vector of normalized unique distances (pdist()) or distance matrix of shape 
        (n_cells, m_cells), before transformation/projection
    post : np.array
        vector of normalized unique distances (pdist()) or distance matrix of shape 
        (n_cells, m_cells), after transformation/projection
    corr_stats : list
        output of `pearsonr()` function correlating the two normalized unique distance 
        vectors
    EMD : float
        output of `wasserstein_1d()` function calculating the Earth Mover's Distance 
        between the two normalized unique distance vectors

    1) performs Pearson correlation of distance distributions
    2) normalizes unique distances using min-max standardization for each dataset
    3) calculates Wasserstein or Earth-Mover's Distance for normalized distance 
    distributions between datasets
    """
    # make sure the number of cells in each matrix is the same
    assert (
        pre.shape == post.shape
    ), 'Matrices contain different number of distances.\n{} in "pre"\n{} in "post"\n'.format(
        pre.shape[0], post.shape[0]
    )

    # if distance matrix (mA x mB, result of cdist), flatten to unique cell-cell distances
    if pre.ndim == 2:
        if verbose:
            print("Flattening pre-transformation distance matrix into 1D array...")
        # if symmetric, only keep unique values (above diagonal)
        if np.allclose(pre, pre.T, rtol=1e-05, atol=1e-08):
            pre = pre[np.triu_indices(n=pre.shape[0], k=1)]
        # otherwise, flatten all distances
        else:
            pre = pre.flatten()

    # if distance matrix (mA x mB, result of cdist), flatten to unique cell-cell distances
    if post.ndim == 2:
        if verbose:
            print("Flattening post-transformation distance matrix into 1D array...")
        # if symmetric, only keep unique values (above diagonal)
        if np.allclose(post, post.T, rtol=1e-05, atol=1e-08):
            post = post[np.triu_indices(n=post.shape[0], k=1)]
        # otherwise, flatten all distances
        else:
            post = post.flatten()

    # if dataset is large, randomly downsample to reasonable number of distances for calculation
    if downsample:
        assert downsample < len(
            pre
        ), "Must provide downsample value smaller than total number of cell-cell distances provided in pre and post"
        if verbose:
            print("Downsampling to {} total cell-cell distances...".format(downsample))
        idx = np.random.choice(np.arange(len(pre)), downsample, replace=False)
        pre = pre[idx]
        post = post[idx]

    # calculate correlation coefficient using Pearson correlation
    if verbose:
        print("Correlating distances")
    corr_stats = pearsonr(x=pre, y=post)

    # min-max normalization for fair comparison of probability distributions
    if verbose:
        print("Normalizing unique distances")
    pre -= pre.min()
    pre /= np.ptp(pre)

    post -= post.min()
    post /= np.ptp(post)

    # calculate EMD for the distance matrices
    # by default, downsample to 50M distances to speed processing time,
    # since this function often breaks with larger distributions
    if verbose:
        print("Calculating Earth-Mover's Distance between distributions")
    if len(pre) > 50000000:
        idx = np.random.choice(np.arange(len(pre)), 50000000, replace=False)
        pre_EMD = pre[idx]
        post_EMD = post[idx]
        EMD = wasserstein_1d(pre_EMD, post_EMD)
    else:
        EMD = wasserstein_1d(pre, post)

    return pre, post, corr_stats, EMD


def joint_plot_distance_correlation(pre, post, labels, save_to, figsize=[6,6], title=None, metric_label=None):
    plt.close()

    cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
    palette = sns.cubehelix_palette(start=.5, rot=-.5)

    fig = sns.JointGrid(x=pre, y=post, space=0, height=figsize[0])
    fig.plot_joint(plt.hist2d, bins=50, cmap=cmap)

    sns.kdeplot(pre,
                color=palette[-1],
                fill=False,
                bw_method=0.01,
                ax=fig.ax_marg_x)
    
    sns.kdeplot(post,
                color=palette[2],
                fill=False,
                bw_method=0.01,
                vertical=True,
                ax=fig.ax_marg_y)
    
    fig.ax_joint.plot(
        np.linspace(max(min(pre), min(post)), 1, 100),
        np.linspace(max(min(pre), min(post)), 1, 100),
        linestyle="dashed",
        color=palette[-1],
    ) 

    if title:
        fig.fig.suptitle(title, fontsize="xx-large", color='black', fontfamily='DejaVu Sans Mono')

    # Add label to top-left corner if provided
    if metric_label:
        fig.ax_joint.text(
            0.05, 0.95,  # X, Y coordinates (0, 0 is bottom left and 1, 1 is top right)
            metric_label,
            transform=fig.ax_joint.transAxes,  # Use Axes coordinates
            fontsize="x-large",
            color='black',
            verticalalignment='top',
            horizontalalignment='left',
            fontfamily='DejaVu Sans Mono'
        )

    plt.xlabel(labels[0], fontsize="xx-large", color=palette[-1], fontfamily='DejaVu Sans Mono')
    plt.ylabel(labels[1], fontsize="xx-large", color=palette[2], fontfamily='DejaVu Sans Mono')

    plt.tick_params(labelbottom=False, labelleft=False)

    plt.tight_layout()
    plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=400)

def plot_cumulative_distribution(pre, post, labels, save_to, figsize=[6,6], legend=True, title=None, metric_label=None):

    num_bins = int(len(pre) / 100)
    pre_counts, pre_bin_edges = np.histogram(pre, bins=num_bins)
    pre_cdf = np.cumsum(pre_counts)
    post_counts, post_bin_edges = np.histogram(post, bins=num_bins)
    post_cdf = np.cumsum(post_counts)

    cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
    palette = sns.cubehelix_palette(start=.5, rot=-.5)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
            pre_bin_edges[1:],
            pre_cdf / pre_cdf[-1],
            label=labels[0],
            color=palette[-1],
        )
    ax.plot(
            post_bin_edges[1:],
            post_cdf / post_cdf[-1],
            label=labels[1],
            color=palette[2],
        )
    
    if legend:
        ax.legend(loc="lower right", 
                  fontsize="xx-large", 
                  prop=dict(family='DejaVu Sans Mono'),
                  title_fontproperties=dict(family='DejaVu Sans Mono')
                  )

    if title:
        fig.suptitle(title, fontsize="xx-large", color='black', fontfamily='DejaVu Sans Mono')

    if metric_label:
        plt.text(
            0.05, 0.95,  # X, Y coordinates (0, 0 is bottom left and 1, 1 is top right)
            metric_label,
            # transform=fig.ax_joint.transAxes,  # Use Axes coordinates
            fontsize="x-large",
            color='black',
            verticalalignment='top',
            horizontalalignment='left',
            fontfamily='DejaVu Sans Mono'
        )

    plt.tight_layout()
    plt.savefig(fname=save_to, transparent=True, bbox_inches="tight", dpi=400)


def compute_wasserstein_distance(latent_vectors_1, latent_vectors_2):
    """
    Compute the Wasserstein distance (optimal transport distance) 
    between two sets of latent vectors.
    
    Parameters:
    - latent_vectors_1: np.ndarray of shape (n_samples, n_features)
        Latent vectors for the first set (e.g., control image).
    - latent_vectors_2: np.ndarray of shape (n_samples, n_features)
        Latent vectors for the second set (e.g., augmented image).
        
    Returns:
    - wasserstein_distance: float
        The computed Wasserstein distance between the two sets of latent vectors.
    """
    
    # Number of samples
    n_samples_1, n_samples_2 = latent_vectors_1.shape[0], latent_vectors_2.shape[0]
    
    # Compute the cost matrix (Euclidean distance)
    cost_matrix = ot.dist(latent_vectors_1, latent_vectors_2)
    
    # Uniform distribution for both sets
    distribution_1 = np.ones(n_samples_1) / n_samples_1
    distribution_2 = np.ones(n_samples_2) / n_samples_2
    
    # Compute Wasserstein distance
    wasserstein_distance = ot.emd2(distribution_1, distribution_2, cost_matrix)
    
    return wasserstein_distance