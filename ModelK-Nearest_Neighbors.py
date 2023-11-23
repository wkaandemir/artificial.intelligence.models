import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,  confusion_matrix
from sklearn.impute import SimpleImputer
import pickle
import os
import hyperparameters
from hyperparameters import K_Nearest_Neighbors_hyperparameters
def run():
    # Load the data
    data = pd.read_excel(hyperparameters.file_path)

    def save_results(result):
        # Output folder path
        output_folder = "output"

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save the result to 'ModelDecisionTreeClassifier.txt' in the output folder
        output_path = os.path.join(output_folder, 'ModelK-Nearest_Neighbors.txt')
        with open(output_path, 'a') as f:
            f.write(result + '\n')

    # Redirect print statements to the save_results function
    print = save_results

    # Separate the target variable and features
    X = data.drop(columns=[hyperparameters.target_column])
    y = data[hyperparameters.target_column]

    # Fill missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter combinations
    param_grid = K_Nearest_Neighbors_hyperparameters

    # Create a K-Nearest Neighbors (KNN) model
    knn_model = KNeighborsClassifier()

    # Use the model and parameter combinations for Grid Search
    grid_search = GridSearchCV(estimator=knn_model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=2)
    grid_search.fit(X_train, y_train)

    # Select the best model
    best_knn_model = grid_search.best_estimator_


    # Path to the models folder
    models_folder = "models"

    # Create the models folder if it doesn't exist
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    # Save the model
    model_path = os.path.join(models_folder, 'ModelK-Nearest_Neighbors.pkl')
    with open(model_path, 'wb') as model_file:
        pickle.dump(best_knn_model, model_file)

    print(f"Model successfully saved to {model_path}.")


    # Make predictions on the test data using the best model
    y_pred = best_knn_model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate the confusion matrix
    confusion = confusion_matrix(y_test, y_pred)

    # Display the classification report
    class_report = classification_report(y_test, y_pred)

    print(f"K-Nearest Neighbors (KNN) Analysis:")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion}")
    print(f"Classification Report:\n{class_report}")
    print(f"Best parameter combination: {grid_search.best_params_}")
    print(f"Best accuracy score: {grid_search.best_score_}")
