from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class KFoldXValidationClassification():
    def __init__(self, k_fold_splits=10):
        # Initialize the model with the number of K-Fold splits
        self.nb_clf_models_kfold = []  # List to store trained models
        self.k_fold_splits = k_fold_splits  # Number of folds for cross-validation
        self.accuracy_score_kfold = 0  # Store the average accuracy score

    def fit(self, X_train, y_train):
        # Initialize the KFold object with the specified number of splits
        k_fold = KFold(n_splits=self.k_fold_splits)
        k_fold_train = k_fold.split(X_train, y_train)  # Generates index pairs for training and test sets

        # Convert input data into pandas Series for easier handling
        X_train = pd.Series(X_train)
        y_train = pd.Series(y_train)

        accuracy_scores = []  # List to store accuracy for each fold
        for (x_indices, y_indices) in k_fold_train:
            # Split the data into training and test sets based on the indices
            X_train_kfold = X_train.iloc[x_indices].tolist()
            X_test_kfold = X_train.iloc[y_indices].tolist()
            y_train_kfold = y_train.iloc[x_indices].tolist()
            y_test_kfold = y_train.iloc[y_indices].tolist()

            # Initialize and train the Naive Bayes classifier (GaussianNB)
            nb_clf = GaussianNB()
            nb_clf.fit(X_train_kfold, y_train_kfold)

            # Make predictions for the test set
            y_pred_test_kfold = nb_clf.predict(X_test_kfold)
            
            # Calculate the accuracy for the current fold and append it to the list
            accuracy = accuracy_score(y_test_kfold, y_pred_test_kfold)
            accuracy_scores.append(accuracy)
            
            # Save the trained model for later voting
            self.nb_clf_models_kfold.append(nb_clf)

        # Calculate the average accuracy over all folds
        self.accuracy_score_kfold = np.mean(accuracy_scores)

    def predict(self, x_new):
        predictions = []  # List to store predictions from each model
        predictions_mean = []  # List to store the final prediction by voting

        # Iterate over all trained models and make predictions for new data
        for model in self.nb_clf_models_kfold:
            predictions_modelk = model.predict(x_new)
            predictions.append(predictions_modelk)

        # For each sample in the new data, aggregate the predictions from the models
        for i in range(len(predictions[0])):
            data_predk = [predictions[j][i] for j in range(len(self.nb_clf_models_kfold))]
            
            # Count the frequency of each prediction (e.g., 'ham' or 'spam')
            unique, counts = np.unique(data_predk, return_counts=True)
            frequency_dict = dict(zip(unique, counts))
            
            # Determine the most frequent prediction (majority vote)
            most_frequent = max(frequency_dict, key=frequency_dict.get)
            predictions_mean.append(most_frequent)
        
        return predictions_mean

    def getAccuracy(self):
        # Return the average accuracy from the K-Fold cross-validation
        return self.accuracy_score_kfold