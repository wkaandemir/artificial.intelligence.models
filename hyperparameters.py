import numpy as np

# Write the name of the target column where it says target.
target_column = "target"
file_path = 'data.xlsx'


# AdaBoost Classifier Hyperparameters for Energy Saving
AdaBoost_hyperparameters = {
    'learning_rate': np.arange(0.01, 0.1, 0.01),
    'n_estimators': range(50, 101, 5)
}

# Decision Tree Classifier Hyperparameters for Energy Saving
DecisionTree_hyperparameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Gradient Boosting Classifier Hyperparameters for Energy Saving
GradientBoosting_hyperparameters = {
    'learning_rate': np.arange(0.01, 0.1, 0.01),
    'max_depth': [3, 5],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 5],
    'n_estimators': range(50, 101, 5),
    'subsample': [0.8, 1.0]
}

# K-Nearest Neighbors Classifier Hyperparameters for Energy Saving
K_Nearest_Neighbors_hyperparameters = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

# Logistic Regression Hyperparameters for Energy Saving
LogisticRegression_hyperparameters = {
    'C': np.arange(0.001, 0.1, 0.01)
}

# Random Forest Classifier Hyperparameters for Energy Saving
RandomForest_hyperparameters = {
    'max_depth': [3, 5],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 5],
    'n_estimators': range(50, 101, 5)
}

# Support Vector Machines Classifier Hyperparameters for Energy Saving
SupportVectorMachines_hyperparameters = {
    "C": np.arange(0.001, 0.1, 0.01),
    "degree": [2, 3],
    "gamma": ['scale', 'auto'],
    "kernel": ['linear', 'rbf']
}

# XGBoost Classifier Hyperparameters for Energy Saving
XGBoost_hyperparameters = {
    'colsample_bytree': [0.8, 1.0],
    'learning_rate': np.arange(0.01, 0.1, 0.01),
    'max_depth': [3, 5],
    'n_estimators': range(50, 101, 5),
    'subsample': [0.8, 1.0]
}
