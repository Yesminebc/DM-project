# Decision Making Project: Clustering and Modeling Customer Preferences

Project Overview :
This project aims to develop models for clustering customers based on their purchasing decisions and identifying the decision-making functions that guide their choices. Understanding customer preferences is crucial for optimizing product offerings in the retail sector. 

The project explores two approaches:
1- Mixed-Integer Programming (MIP) Solution: A mathematical optimization model that clusters customers using the UTA (Utility Additive) model.

2- Heuristic Solutions: Scalable methods including K-Means clustering and a preference-based Soft-KMeans algorithm for handling large datasets efficiently.

# Installation Instructions
Run the following command to install the required libraries:
```Bash
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install gurobipy
pip install torch
pip install torchvision
```

If you encounter issues with gurobipy, ensure you have a valid Gurobi license.
For visualization, matplotlib and seaborn are used.

# Files Description

data.py: Implements the Dataloader class to load datasets from .npy files, including customer preferences and ground truth labels.

metrics.py: Defines evaluation metrics such as PairsExplained and ClusterIntersection to assess clustering performance.

2 models.py files : Define database models and their attributes. Each file has :
- The class TwoClustersMIP (Mixed-Integer Programming Model)
- The class HeuristicModel containning one of the two proposed heuristic solutions.


evaluation.py : evaluates the performance of the two models.

# Authors

Xuanchong CHEN

Haonan ZHU

Yasmine Ben Cheikh

# Supervisor

Mr. Vincent Mousseau
