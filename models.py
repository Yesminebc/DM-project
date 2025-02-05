import pickle
from gurobipy import GRB, quicksum
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import visualize_clusters, PreferenceSoftKMeans
import numpy as np
import torch
import torch.nn as nn

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
        env.setParam('TimeLimit', 100)  # Time limit in seconds
        env.setParam('MIPFocus', 3)  # Prioritize reducing the number of nodes in the search tree
        env.setParam('MIPGap', 1e-2)
        env.setParam('OutputFlag', 1)  # Enable output of detailed log

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
        # Define the postive and negative errors
        self.errors_positive = self.model.addVars(n_samples,self.n_clusters, 2, vtype=GRB.CONTINUOUS, name="errors_positive")
        self.errors_negative = self.model.addVars(n_samples,self.n_clusters, 2, vtype=GRB.CONTINUOUS, name="errors_negative")

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.instantiate(n_features, n_samples)

        feature_maxs_X,feature_mins_X = np.max(X, axis=0),np.min(X, axis=0)
        feature_maxs_Y,feature_mins_Y = np.max(Y, axis=0),np.min(Y, axis=0)
        # For the overlapping range:
        # The global lower bound for each feature is the larger of the two minimums
        global_feature_mins = np.maximum(feature_mins_X, feature_mins_Y)
        # The global upper bound for each feature is the smaller of the two maximums
        global_feature_maxs = np.minimum(feature_maxs_X, feature_maxs_Y)
        # Now, global_feature_mins and global_feature_maxs contain the overlapping range for each feature.
        print("Overlapping feature lower bounds:", global_feature_mins)
        print("Overlapping feature upper bounds:", global_feature_maxs)

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
                                self.feature_utility_Y[i, k, j] == self.p[j, k] * (
                                    (
                                    self.utility_margin[j, k, seg] +
                                    (self.utility_margin[j, k, seg + 1] - self.utility_margin[j, k, seg]) *
                                    (Y[i, j] - seg_start) / seg_diff )
                                )
                            )

                    self.model.addConstr(
                        self.utility_X[i, k] == quicksum(self.feature_utility_X[i, k, j] for j in range(n_features))- self.errors_positive[i,k,0] + self.errors_negative[i,k,0]
                    )
                    self.model.addConstr(
                        self.utility_Y[i, k] == quicksum(self.feature_utility_Y[i, k, j] for j in range(n_features)) - self.errors_positive[i,k,1] + self.errors_negative[i,k,1]
                    )


        epsilons = 0.05
        M = 1.1

        for i in range(n_samples):
            for k in range(self.n_clusters):
                self.model.addConstr(
                    self.utility_X[i, k] - self.utility_Y[i, k]  +
                    (1 - self.clusters_assignments[i, k]) * M >= epsilons
                )
            self.model.addConstr(
                quicksum(self.clusters_assignments[i, k] for k in range(self.n_clusters)) >= 1
            )

        for k in range(self.n_clusters):
            for j in range(n_features):
                for seg in range(self.n_pieces):
                    self.model.addConstr(
                        self.utility_margin[j, k, seg + 1] >= self.utility_margin[j, k, seg] + epsilons
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
            quicksum(self.errors_positive[i,k,seg] + self.errors_negative[i,k,seg] for i in range(n_samples) for k in range(self.n_clusters) for seg in range(2)),
            GRB.MINIMIZE)

        print("Starting optimization...")
        self.model.optimize()
        print("Optimization complete.")
        # After clustering is done
        data = np.hstack((X, Y))
        cluster_assignments = np.array([
            np.argmax([self.clusters_assignments[i, k].X for k in range(self.n_clusters)])
            for i in range(n_samples)
        ])
        visualize_clusters(data, cluster_assignments, method='PCA')

        print("Utility margins:")
        for j in range(n_features):
            for k in range(self.n_clusters):
                margins = [self.utility_margin[j, k, seg].X for seg in range(self.n_pieces + 1)]
                print(f"Feature {j + 1}, Cluster {k + 1}: {margins}")

        print(
            f"Cluster assignments: {[self.clusters_assignments[i, k].X for i in range(n_samples) for k in range(self.n_clusters)]}")


        print("Feature weights (p):")
        for j in range(n_features):
            for k in range(self.n_clusters):
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

class PreferenceModel(nn.Module):
    def __init__(self, input_dim):
        super(PreferenceModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出效用值
        )

    def forward(self, x, y):
        """
        simultaneously compute the utility values for X and Y.

        Parameters:
        -----------
        x: torch.Tensor
        y: torch.Tensor

        Returns:
        --------
        utility_x: torch.Tensor
            U(X)
        utility_y: torch.Tensor
            U(Y)
        """
        utility_x = self.network(x)
        utility_y = self.network(y)
        return utility_x, utility_y

def preference_loss(utility_X, utility_Y, epsilon=0.01, weight_decay=0.01):
    preference_term = torch.relu(epsilon - (utility_X - utility_Y))
    penalty_term = weight_decay * torch.mean((utility_X - utility_Y) ** 2)
    smoothness_term = torch.mean(torch.abs(utility_X[1:] - utility_X[:-1]))
    return torch.mean(preference_term) + penalty_term + 0.1 * smoothness_term

class ClusterUtilityPredictor:
    def __init__(self, n_clusters, input_dim):
        self.n_clusters = n_clusters
        self.models = {i: PreferenceModel(input_dim) for i in range(n_clusters)}
        self.training_results = {i: {"losses": [], "satisfaction_ratios": []} for i in range(n_clusters)}

    def train_for_clusters(self, grouped_X, grouped_Y, epsilon=0.01, epochs=50, batch_size=16, learning_rate=0.001):
        """
        Indepedently train a utility predictor for each cluster.

        Parameters:
        -----------
        grouped_X: dict
            Preffered samples feature matrix grouped by cluster.
        grouped_Y: dict
            Unchosen samples feature matrix grouped by cluster.
        epsilon: float
            Minimum utility difference between X and Y.
        epochs: int
            Number of training epochs.
        batch_size: int

        learning_rate: float

        """
        for cluster_id in range(self.n_clusters):
            X_cluster = grouped_X.get(cluster_id, [])
            Y_cluster = grouped_Y.get(cluster_id, [])

            if len(X_cluster) == 0:
                print(f"Cluster {cluster_id} has no data, skipping.")
                continue

            # Transfer to numpy arrays
            X_cluster = np.array(X_cluster)
            Y_cluster = np.array(Y_cluster)

            print(f"Training model for Cluster {cluster_id} with {len(X_cluster)} samples...")

            # Train the model by train_with_preference_loss
            model = self.models[cluster_id]
            losses = []
            satisfaction_ratios = []

            features_X = torch.tensor(X_cluster, dtype=torch.float32)
            features_Y = torch.tensor(Y_cluster, dtype=torch.float32)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            dataset = torch.utils.data.TensorDataset(features_X, features_Y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(epochs):
                total_loss = 0
                correct_preferences = 0
                total_samples = 0

                for batch_X, batch_Y in dataloader:
                    optimizer.zero_grad()

                    # calculate the utility values for X and Y
                    utility_X, utility_Y = model(batch_X, batch_Y)

                    # preference loss
                    loss = preference_loss(utility_X, utility_Y, epsilon)
                    total_loss += loss.item()

                    # calculate the number of satisfied preferences
                    satisfied = (utility_X > utility_Y + epsilon).float()
                    correct_preferences += satisfied.sum().item()
                    total_samples += len(satisfied)

                    # backpropagation and optimization
                    loss.backward()
                    optimizer.step()

                # record the training results at the end of each epoch
                satisfaction_ratio = correct_preferences / total_samples
                satisfaction_ratios.append(satisfaction_ratio)
                losses.append(total_loss)

                print(f"Cluster {cluster_id}, Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, "
                      f"Satisfaction Ratio: {satisfaction_ratio:.4f}")

            # save the training results
            self.training_results[cluster_id]["losses"] = losses
            self.training_results[cluster_id]["satisfaction_ratios"] = satisfaction_ratios

    def plot_training_results(self):
        """
        Draw the loss and satisfaction ratio curves for each cluster.
        """
        import matplotlib.pyplot as plt

        for cluster_id, results in self.training_results.items():
            losses = results["losses"]
            satisfaction_ratios = results["satisfaction_ratios"]

            if not losses or not satisfaction_ratios:
                print(f"Cluster {cluster_id} has no training data, skipping.")
                continue

            epochs = range(1, len(losses) + 1)

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(epochs, losses, label=f"Cluster {cluster_id} Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"Cluster {cluster_id} Loss Curve")
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(epochs, satisfaction_ratios, label=f"Cluster {cluster_id} Satisfaction", color="orange")
            plt.xlabel("Epochs")
            plt.ylabel("Satisfaction Ratio")
            plt.title(f"Cluster {cluster_id} Satisfaction Curve")
            plt.legend()

            plt.tight_layout()
            plt.show()

    def predict_utilities(self, X):
        """
        predict the utility values for each sample in each cluster.

        Parameters:
        -----------
        X: np.ndarray
           test samples feature matrix.

        Returns:
        --------
        utilities: np.ndarray
            The utility values for each sample in each cluster.
        """
        n_samples = X.shape[0]
        utilities = np.zeros((n_samples, self.n_clusters))

        for cluster_id, model in self.models.items():
            # ensure the model is in evaluation mode
            device = next(model.parameters()).device
            features_X = torch.tensor(X, dtype=torch.float32).to(device)
            utilities[:, cluster_id] = model.network(features_X).detach().cpu().numpy().flatten()

        return utilities


class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_clusters, max_iter=1000, random_state=123):
        """Initialization of the Heuristic Model.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.cluster_assignments = None
        self.utility_function = [None] * n_clusters
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

    def instantiate(self):
        """Instantiation of the clustering model."""
        return

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
        # preference constraints
        preference_constraints = np.where((X - Y).sum(axis=1, keepdims=True) > 0, 1.0, 0.5)
        # clustering
        delta = X - Y
        cluster_data = delta
        data_scaled = self.scaler_X.fit_transform(cluster_data)
        soft_kmeans = PreferenceSoftKMeans(n_clusters=3, beta=2.0)
        soft_kmeans.fit(cluster_data, preference_constraints)
        self.cluster_assignments = np.argmax(soft_kmeans.predict(data_scaled), axis=1)

        print("data is assigned in the following way", self.cluster_assignments)
        df = pd.DataFrame({"Cluster": self.cluster_assignments})
        filename = "cluster_assignments.csv"
        df.to_csv(filename, index=False)
        print(f"Cluster assignments saved to {filename}")
        # After clustering is done
        visualize_clusters(data_scaled, self.cluster_assignments, method='PCA')
        # group the data by cluster
        clusters = {i: [] for i in range(self.n_clusters)}
        for idx, cluster in enumerate(self.cluster_assignments):
            clusters[cluster].append((X[idx], Y[idx]))

        self.cluster_assignments = np.argmax(soft_kmeans.predict(cluster_data), axis=1)

        grouped_X = {i: [] for i in range(self.n_clusters)}
        grouped_Y = {i: [] for i in range(self.n_clusters)}

        for idx, cluster_id in enumerate(self.cluster_assignments):
            grouped_X[cluster_id].append(X[idx])
            grouped_Y[cluster_id].append(Y[idx])

        for cluster_id in range(self.n_clusters):
            grouped_X[cluster_id] = np.array(grouped_X[cluster_id])
            grouped_Y[cluster_id] = np.array(grouped_Y[cluster_id])

        utility_predictor = ClusterUtilityPredictor(self.n_clusters, input_dim=n_features)
        utility_predictor.train_for_clusters(
            grouped_X, grouped_Y, epsilon=0.01, epochs=10, batch_size=16, learning_rate=0.0001
        )
        utility_predictor.plot_training_results()
        self.utility_function = utility_predictor
        return

    def predict_utility(self, X):
        """Return Decision Function of the Heuristic Model for X.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns:
        --------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # call
        utilities = self.utility_function.predict_utilities(X)
        return utilities

        return utilities
