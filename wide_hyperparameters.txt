import numpy as np

# Write the name of the target column where it says target.
target_column = "target"
file_path = 'data.xlsx'


# AdaBoost Classifier Hyperparameters
AdaBoost_hyperparameters = {
    'learning_rate': np.arange(0.1, 1.1, 0.1),  # Typical range for learning rates
    'n_estimators': range(50, 201, 10)  # A reasonable range for the number of trees
}

# Decision Tree Classifier Hyperparameters
DecisionTree_hyperparameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Gradient Boosting Classifier Hyperparameters
GradientBoosting_hyperparameters = {
    'learning_rate': [0.01, 0.1, 0.2],  # Common learning rates
    'max_depth': [3, 4, 5],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 5],
    'n_estimators': [100, 150, 200],  # A reasonable range for the number of trees
    'subsample': [0.8, 0.9, 1.0]
}

# K-Nearest Neighbors Classifier Hyperparameters
K_Nearest_Neighbors_hyperparameters = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

# Logistic Regression Hyperparameters
LogisticRegression_hyperparameters = {
    'C': [0.001, 0.01, 0.1, 1.0, 10.0]
}

# Random Forest Classifier Hyperparameters
RandomForest_hyperparameters = {
    'max_depth': [None, 5, 10, 20],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [100, 150, 200]  # A reasonable range for the number of trees
}

# Support Vector Machines Classifier Hyperparameters
SupportVectorMachines_hyperparameters = {
    "C": [0.001, 0.01, 0.1, 1.0],
    "degree": [2, 3],
    "gamma": ['scale', 'auto'],
    "kernel": ['linear', 'rbf']
}

# XGBoost Classifier Hyperparameters
XGBoost_hyperparameters = {
    'colsample_bytree': [0.8, 0.9, 1.0],
    'learning_rate': [0.01, 0.1, 0.2],  # Common learning rates
    'max_depth': [3, 4, 5],
    'n_estimators': [100, 150, 200],  # A reasonable range for the number of trees
    'subsample': [0.8, 0.9, 1.0]
}
