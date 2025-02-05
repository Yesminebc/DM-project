from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def gmm_clustering(data_scaled, n_clusters):
    """
    cluster the data using Gaussian Mixture Model (GMM).

    Parameters:
    -----------
    data_scaled: np.ndarray
        standardized data (shape: n_samples x n_features).
    n_clusters: int
        number of clusters.

    Returns:
    --------
    cluster_assignments: np.ndarray
        cluster_assignments[i] means the cluster index of the i-th sample.
    gmm_model: GaussianMixture
        trained GMM model.
    """
    # initialize GMM and fit the model
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm.fit(data_scaled)
    cluster_assignments = gmm.predict(data_scaled)

    # 可视化聚类结果
    plt.figure(figsize=(8, 6))
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=cluster_assignments, cmap='viridis', s=10)
    plt.colorbar()
    plt.title("GMM Clustering Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    return cluster_assignments, gmm
def visualize_clusters(data_scaled, cluster_assignments, method='PCA', n_components=2):
    """
    visualize the clustering results in 2D space using PCA or t-SNE.

    Parameters:
    -----------
    data_scaled: np.ndarray
        standardized data (shape: n_samples x n_features).
    cluster_assignments: np.ndarray
        result of clustering (shape: n_samples).
    method: str
        method for dimensionality reduction ('PCA' or 't-SNE').
    n_components: int
        obtained number of components after dimensionality reduction.(default: 2)

    Returns:
    --------
    None
    """
    if method == 'PCA':
        reducer = PCA(n_components=n_components)
        title = "PCA Visualization of Clusters"
    elif method == 't-SNE':
        reducer = TSNE(n_components=n_components, random_state=42)
        title = "t-SNE Visualization of Clusters"
    else:
        raise ValueError("Unsupported method. Choose 'PCA' or 't-SNE'.")

    reduced_data = reducer.fit_transform(data_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_assignments, cmap='viridis', s=10)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid()
    plt.show()

import numpy as np
from sklearn.cluster import KMeans

class SoftKMeans:
    def __init__(self, n_clusters=3, beta=2.0):
        self.n_clusters = n_clusters
        self.beta = beta

    def fit(self, X):
        """
        cluster the data using soft k-means algorithm.
        """
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=20)
        self.kmeans.fit(X)
        self.cluster_centers_ = self.kmeans.cluster_centers_

        # calculate the responsibilities
        distances = np.linalg.norm(X[:, None] - self.cluster_centers_, axis=2)
        exp_distances = np.exp(-self.beta * distances)
        soft_assignments = exp_distances / exp_distances.sum(axis=1, keepdims=True)

        # hard assignments
        cluster_assignments = np.argmax(soft_assignments, axis=1)

        return cluster_assignments


class PreferenceSoftKMeans:
    def __init__(self, n_clusters=3, beta=2.0):
        self.n_clusters = n_clusters
        self.beta = beta

    def fit(self, X, preference_constraints=None):
        # Step 1: initialize cluster centers
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=20)
        self.kmeans.fit(X)
        self.cluster_centers_ = self.kmeans.cluster_centers_

        # Step 2: iterate to update cluster centers
        for iteration in range(10):  # num_iterations
            distances = np.linalg.norm(X[:, None] - self.cluster_centers_, axis=2)
            exp_distances = np.exp(-self.beta * distances)
            responsibilities = exp_distances / exp_distances.sum(axis=1, keepdims=True)

            # incorporate preference constraints
            if preference_constraints is not None:
                responsibilities *= preference_constraints

            # update cluster centers
            self.cluster_centers_ = np.dot(responsibilities.T, X) / responsibilities.sum(axis=0)[:, None]

    def predict(self, X):
        distances = np.linalg.norm(X[:, None] - self.cluster_centers_, axis=2)
        exp_distances = np.exp(-self.beta * distances)
        return exp_distances / exp_distances.sum(axis=1, keepdims=True)



