from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def gmm_clustering(data_scaled, n_clusters):
    """
    使用 GMM 进行聚类，并返回聚类结果和模型。

    Parameters:
    -----------
    data_scaled: np.ndarray
        标准化后的数据。
    n_clusters: int
        聚类数。

    Returns:
    --------
    cluster_assignments: np.ndarray
        聚类分配结果。
    gmm_model: GaussianMixture
        训练好的 GMM 模型。
    """
    # 初始化并训练 GMM 模型
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
    使用 PCA 或 t-SNE 对聚类结果进行降维可视化。

    Parameters:
    -----------
    data_scaled: np.ndarray
        标准化后的数据（形状为 n_samples x n_features）。
    cluster_assignments: np.ndarray
        聚类分配结果（形状为 n_samples）。
    method: str
        降维方法 ('PCA' 或 't-SNE')。
    n_components: int
        降维的目标维度（通常为 2）。

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
import numpy as np
from sklearn.cluster import KMeans

# Soft KMeans 聚类
class SoftKMeans:
    def __init__(self, n_clusters=3, beta=2.0):
        self.n_clusters = n_clusters
        self.beta = beta

    def fit(self, X):
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=20)
        self.kmeans.fit(X)
        self.cluster_centers_ = self.kmeans.cluster_centers_

    def predict(self, X):
        distances = np.linalg.norm(X[:, None] - self.cluster_centers_, axis=2)
        exp_distances = np.exp(-self.beta * distances)
        return exp_distances / exp_distances.sum(axis=1, keepdims=True)

class PreferenceSoftKMeans:
    def __init__(self, n_clusters=3, beta=2.0):
        self.n_clusters = n_clusters
        self.beta = beta

    def fit(self, X, preference_constraints=None):
        # Step 1: 初始化
        self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=20)
        self.kmeans.fit(X)
        self.cluster_centers_ = self.kmeans.cluster_centers_

        # Step 2: 迭代更新
        for iteration in range(10):  # 迭代次数
            distances = np.linalg.norm(X[:, None] - self.cluster_centers_, axis=2)
            exp_distances = np.exp(-self.beta * distances)
            responsibilities = exp_distances / exp_distances.sum(axis=1, keepdims=True)

            # 引入偏好约束
            if preference_constraints is not None:
                responsibilities *= preference_constraints

            # 更新聚类中心
            self.cluster_centers_ = np.dot(responsibilities.T, X) / responsibilities.sum(axis=0)[:, None]

    def predict(self, X):
        distances = np.linalg.norm(X[:, None] - self.cluster_centers_, axis=2)
        exp_distances = np.exp(-self.beta * distances)
        return exp_distances / exp_distances.sum(axis=1, keepdims=True)


from sklearn.linear_model import LogisticRegression
from scipy.special import expit


