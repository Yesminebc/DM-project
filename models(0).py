import pickle
import numpy as np
from gurobipy import GRB, Model, quicksum
from scipy.optimize import minimize

class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
        pass


    def fit(self, X, Y):
        """Fit function to find the parameters according to (X, Y) data.
        (X, Y) formatting must be so that X[i] is preferred to Y[i] for all i.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # Customize what happens in the fit function
        return

    def predict_utility(self, X):
        """Method to call the decision function of your model

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # Customize what happens in the predict utility function
        return

    def predict_preference(self, X, Y):
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return (X_u - Y_u > 0).astype(int)

    def predict_cluster(self, X, Y):
        """Predict which cluster prefers X over Y THE MOST, meaning that if several cluster prefer X over Y, it will
        be assigned to the cluster showing the highest utility difference). The reversal is True if none of the clusters
        prefer X over Y.
        Compared to predict_preference, it indicates a cluster index.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, ) index of cluster with highest preference difference between X and Y.
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return np.argmax(X_u - Y_u, axis=1)

    def save_model(self, path):
        """Save the model in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the file in which the model will be saved
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(clf, path):
        """Load a model saved in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the path to the file to load
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model

class TwoClustersMIP(BaseModel):
    def __init__(self, n_pieces, n_clusters):
        self.breakpoints = None
        self.model = None
        self.n_pieces = n_pieces
        self.n_clusters = n_clusters

    def instantiate(self, n_features, n_samples):
        from gurobipy import Env, Model
        env = Env()
        env.setParam('Threads', 8)  # Limit the number of threads to 8
        env.setParam('Method', 2)  # Use Barrier method
        env.setParam('ConcurrentMIP', 1)  # Enable concurrent MIP
        # env.setParam('OutputFlag', 0)  # 禁用详细日志
        env.setParam('TimeLimit', 300)  # 设置时间限制为 300 秒
        env.setParam('MIPFocus', 3)  # 优先找可行解
        env.setParam('MIPGap', 1e-2)
        env.setParam('OutputFlag', 1)  # 启用详细日志

        self.model = Model("MIP_Preferences", env=env)

        self.utility_margin = self.model.addVars(
            n_features, self.n_clusters, self.n_pieces+1,
            lb=0, vtype=GRB.CONTINUOUS, name="utility_margin"
        )

        self.utility_X = self.model.addVars(
            n_samples, self.n_clusters,
            lb=0, vtype=GRB.CONTINUOUS, name="utility_X"
        )

        self.utility_Y = self.model.addVars(
            n_samples, self.n_clusters,
            lb=0, vtype=GRB.CONTINUOUS, name="utility_Y"
        )

        self.feature_utility_X = self.model.addVars(
            n_samples, self.n_clusters, n_features,
            lb=0, vtype=GRB.CONTINUOUS, name="feature_utility_X"
        )

        self.feature_utility_Y = self.model.addVars(
            n_samples, self.n_clusters, n_features,
            lb=0, vtype=GRB.CONTINUOUS, name="feature_utility_Y"
        )

        self.clusters_assignments = self.model.addVars(
            n_samples, self.n_clusters,
            vtype=GRB.BINARY, name="clusters_assignments"
        )

        self.errors = self.model.addVars(
            self.n_clusters, n_samples,
            lb=0, vtype=GRB.CONTINUOUS, name="errors"
        )
        self.p = self.model.addVars(
            n_features, self.n_clusters,
            lb=0, ub=1, vtype=GRB.CONTINUOUS, name="p"
        )

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.instantiate(n_features, n_samples)

        # x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
        self.breakpoints = [
            np.linspace(0, 1, self.n_pieces + 1) for _ in range(n_features)
        ]

        # breakpoint_diffs = np.array([
        #     self.breakpoints[j][seg + 1] - self.breakpoints[j][seg]
        #     for j in range(n_features) for seg in range(self.n_pieces)
        # ]).reshape(n_features, self.n_pieces)

        breakpoint_diffs = np.full((n_features, self.n_pieces), 1 / self.n_pieces)

        for k in range(self.n_clusters):
            for i in range(n_samples):
                for j in range(n_features):
                    for seg in range(self.n_pieces):
                        seg_start = self.breakpoints[j][seg]
                        seg_end = self.breakpoints[j][seg + 1]
                        # seg_diff = breakpoint_diffs[j, seg]
                        seg_diff = 0.2
                        if seg_start < X[i, j] < seg_end:
                            self.model.addConstr(
                                self.feature_utility_X[i, k, j] == self.p[j, k] * ((
                                    self.utility_margin[j, k, seg] +
                                    (self.utility_margin[j, k, seg + 1] - self.utility_margin[j, k, seg]) *
                                    (X[i, j] - seg_start) / seg_diff)
                                )
                            )
                        if seg_start < Y[i, j] < seg_end:
                            self.model.addConstr(
                                self.feature_utility_Y[i, k, j] == self.p[j, k] * ((
                                    self.utility_margin[j, k, seg] +
                                    (self.utility_margin[j, k, seg + 1] - self.utility_margin[j, k, seg]) *
                                    (self.breakpoints[j][seg] - seg_start) / seg_diff )
                                )
                            )

                    self.model.addConstr(
                        self.utility_X[i, k] == quicksum(self.feature_utility_X[i, k, j] for j in range(n_features))
                    )
                    self.model.addConstr(
                        self.utility_Y[i, k] == quicksum(self.feature_utility_Y[i, k, j] for j in range(n_features))
                    )

        # epsilons = np.percentile(X - Y, 75)
        epsilons = 0.3
        M = 1000
        # M = 10 * (np.max(X) - np.min(X))


        for i in range(n_samples):
            for k in range(self.n_clusters):
                self.model.addConstr(
                    self.utility_X[i, k] - self.utility_Y[i, k] + self.errors[k, i] +
                    (1 - self.clusters_assignments[i, k]) * M >= epsilons
                )
            self.model.addConstr(
                quicksum(self.clusters_assignments[i, k] for k in range(self.n_clusters)) >= 1
            )

        for k in range(self.n_clusters):
            for j in range(n_features):
                for seg in range(self.n_pieces):
                    self.model.addConstr(
                        self.utility_margin[j, k, seg + 1] >= self.utility_margin[j, k, seg]
                    )
                self.model.addConstr(
                    self.utility_margin[j, k, 0] == 0
                )
                self.model.addConstr(
                    self.utility_margin[j, k, self.n_pieces] == 1
                )
            self.model.addConstr(
                quicksum(self.p[j, k] for j in range(n_features)) == 1
            )
        # objective
        self.model.setObjective(
            quicksum(self.errors[k, i] for i in range(n_samples) for k in range(self.n_clusters)),
            GRB.MINIMIZE
        )

        print("Starting optimization...")
        self.model.optimize()
        print("Optimization complete.")
        # print(
        #     f"Utility margins: {[self.utility_margin[j, k, seg].X for j in range(n_features) for k in range(self.n_clusters) for seg in range(self.n_pieces)]}")
        # 格式化输出 Utility margins
        print("Utility margins:")
        for j in range(n_features):  # 遍历特征
            for k in range(self.n_clusters):  # 遍历类
                margins = [self.utility_margin[j, k, seg].X for seg in range(self.n_pieces + 1)]
                print(f"Feature {j + 1}, Cluster {k + 1}: {margins}")

        print(
            f"Cluster assignments: {[self.clusters_assignments[i, k].X for i in range(n_samples) for k in range(self.n_clusters)]}")
        # print(
        #     f"p: {[self.p[j, k].X for j in range(n_features) for k in range(self.n_clusters)]}"
        # )
        # 格式化输出 p
        print("Feature weights (p):")
        for j in range(n_features):  # 遍历特征
            for k in range(self.n_clusters):  # 遍历类
                p_value = self.p[j, k].X
                print(f"Feature {j + 1}, Cluster {k + 1}: {p_value}")

    def predict_utility(self, X):
        n_samples, n_features = X.shape
        utility = np.zeros((n_samples, self.n_clusters))

        for k in range(self.n_clusters):
            for i in range(n_samples):
                for j in range(n_features):
                    for seg in range(self.n_pieces):
                        if self.breakpoints[j][seg] < X[i, j] < self.breakpoints[j][seg + 1]:
                            margin_start = self.utility_margin[j, k, seg].X
                            margin_end = self.utility_margin[j, k, seg + 1].X
                            p = self.p[j, k].X
                            utility[i, k] += p * (
                                margin_start + (margin_end - margin_start) *
                                (X[i, j] - self.breakpoints[j][seg]) /
                                (self.breakpoints[j][seg + 1] - self.breakpoints[j][seg])
                            )

        return utility

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_clusters=2, max_iter=100 ):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123 # for reproducibility
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.cluster_assignments = None
        self.model = self.instantiate()
        self.utility_function = [0,0]
        
    def instantiate(self):
        """Instantiation of the clustering model."""
        # Initialize the KMeans model with the desired number of clusters and a fixed random state
        return KMeans(n_clusters=self.n_clusters, random_state=self.seed)

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        n_samples, n_features = X.shape
        # Combine X and Y into a single dataset for clustering
        data = np.hstack((X, Y))

        # Initial clustering using KMeans
        self.model.fit(data)
        self.cluster_assignments = self.model.fit_predict(data)
        self.cluster_centers_ = self.model.cluster_centers_
        
        for iteration in range(self.max_iter):
            print(f"Iteration {iteration + 1}")
            utilities_X = np.zeros((n_samples, self.n_clusters))
            utilities_Y = np.zeros((n_samples, self.n_clusters))
            for cluster in range(self.n_clusters):
                # Find indices of the samples currently assigned to the cluster
                cluster_indices = np.where(self.cluster_assignments == cluster)[0]
                
                if len(cluster_indices) > 0:
                    #extract the X and Y values in the cluster
                    X_cluster = X[cluster_indices]
                    Y_cluster = Y[cluster_indices]
                    self.utility_function[cluster] = np.mean(X_cluster - Y_cluster, axis=0)
                    # Calculate the utilities for X and Y separately
                    utilities_X[:, cluster] = np.dot(X - Y, self.utility_function[cluster])  # Utility for X and Y combined
                    utilities_Y[:, cluster] = np.dot(Y - X, self.utility_function[cluster])  # Utility for Y and X combined

            new_cluster_assignments = np.argmax(utilities_X-utilities_Y, axis=1)
            if np.array_equal(new_cluster_assignments, self.cluster_assignments):
                break

            self.cluster_assignments = new_cluster_assignments
            

    def predict_utility(self, X):
        """Return Decision Function of the Heuristic Model for X.

            Parameters:
            -----------
            X: np.ndarray
            (n_samples, n_features) list of features of elements

            Returns
            -------
            np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
            """          
        utilities = np.zeros((X.shape[0], self.n_clusters))
        for cluster in range(self.n_clusters):
            if self.utility_function[cluster] is not None:
                # compute utility for X
                utilities[:, cluster] = np.dot(X, self.utility_function[cluster])
        return utilities
