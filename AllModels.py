import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, train_test_split, learning_curve
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from hyperparameters import (
    AdaBoost_hyperparameters,
    GradientBoosting_hyperparameters,
    RandomForest_hyperparameters,
    K_Nearest_Neighbors_hyperparameters,
    LogisticRegression_hyperparameters,
    SupportVectorMachines_hyperparameters,
    XGBoost_hyperparameters,
    DecisionTree_hyperparameters
)
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix, classification_report
import joblib

# File path and target column name
file_path = 'data.xlsx' # Place the data file next to this script and specify its name here
target_column = "target" # Enter the target column name

# Load the data
data = pd.read_excel(file_path)

# Sample data loading or creation
X = data.drop(columns=[target_column])
y = data[target_column]

# Define and train models
class ModelAnalysis:
    def __init__(self, model_name, model_class, hyperparameters):
        self.model_name = model_name
        self.model_class = model_class
        self.hyperparameters = hyperparameters
        self.best_params = None
        self.best_score = None
        self.model = None

    def train(self):
        print(f"Training {self.model_name}...")
        model = self.model_class()
        random_search = RandomizedSearchCV(estimator=model, param_distributions=self.hyperparameters, n_iter=50, cv=3, verbose=2, random_state=42, n_jobs=-1)
        random_search.fit(X, y)
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_

    def save_model(self):
        model_filename = f"AllModelsPKL/Model{self.model_name.replace(' ', '')}.pkl"
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        joblib.dump(self.model, model_filename)
        print(f"Model successfully saved to {model_filename}.")

    def evaluate(self):
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_pred)
        brier_score = brier_score_loss(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        cr = classification_report(y, y_pred)

        output_filename = f"AllModelsTXT/{self.model_name.replace(' ', '')}.txt"
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, 'w') as f:
            f.write(f"{self.model_name} Analysis:\n")
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"ROC-AUC: {roc_auc}\n")
            f.write(f"Brier Score: {brier_score}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")
            f.write(f"Classification Report:\n{cr}\n")
            f.write(f"Best parameter combination: {self.best_params}\n")
            f.write(f"Best accuracy score: {self.best_score}\n")
        print(f"Results saved to {output_filename}")

    def plot_learning_curve(self):
        print(f"Plotting learning curve for {self.model_name}...")
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X, y, cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure()
        plt.title(f"Learning Curve: {self.model_name}")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        plot_dir = "AllModelsPlots"
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/LearningCurve_{self.model_name.replace(' ', '')}.png")
        plt.close()
        print(f"Learning curve saved to {plot_dir}/LearningCurve_{self.model_name.replace(' ', '')}.png")

# Create and train models
models = [
    ModelAnalysis('AdaBoost Classifier', AdaBoostClassifier, AdaBoost_hyperparameters),
    ModelAnalysis('Gradient Boosting Classifier', GradientBoostingClassifier, GradientBoosting_hyperparameters),
    ModelAnalysis('Random Forest Classifier', RandomForestClassifier, RandomForest_hyperparameters),
    ModelAnalysis('K-Nearest Neighbors Classifier', KNeighborsClassifier, K_Nearest_Neighbors_hyperparameters),
    ModelAnalysis('Logistic Regression', LogisticRegression, LogisticRegression_hyperparameters),
    ModelAnalysis('Support Vector Machines Classifier', SVC, SupportVectorMachines_hyperparameters),
    ModelAnalysis('XGBoost Classifier', XGBClassifier, XGBoost_hyperparameters),
    ModelAnalysis('Decision Tree Classifier', DecisionTreeClassifier, DecisionTree_hyperparameters)
]

for model in models:
    model.train()
    model.save_model()
    model.evaluate()
    model.plot_learning_curve()
