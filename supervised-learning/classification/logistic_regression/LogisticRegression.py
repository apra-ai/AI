## Packages
import pandas as pd
import numpy as np

# Define a Logistic Regression class with gradient descent
class LogisticRegression():
    def __init__(self, learning_rate=0.0001, max_iter=40, err=1e-3):
        """
        Initializes the logistic regression model with specified hyperparameters.

        Parameters:
        learning_rate: The rate at which the model learns; higher values can speed up convergence but risk instability.
        max_iter: Maximum number of iterations for gradient descent.
        err: Convergence tolerance; training will stop if the gradient's mean squared error is below this value.
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.err = err
        self.param = []  # Initialize parameter vector
    
    def fit(self, X, Y):
        """
        Fits the logistic regression model to the data using gradient descent.

        Parameters:
        X: Training data matrix (features).
        Y: Target labels for training data.
        """
        size_data = X.shape[1]  # Number of features
        
        # Initialize parameter vector to ones
        self.param = [1 for _ in range(size_data)]

        # Gradient descent loop
        for _ in range(self.max_iter):
            # Make predictions on training data
            prediction = self.predict(X)
            
            # Compute gradient
            grad = np.dot(np.subtract(prediction, Y), X) / len(Y)
            
            # Update parameters using the learning rate
            self.param = np.subtract(self.param, self.learning_rate * grad)
            
            # Check for convergence based on gradient's mean squared error
            if np.mean(grad**2) < self.err:
                break

    def predict(self, X):
        """
        Predicts binary class labels for input data X.

        Parameters:
        X: Data matrix for which predictions are made.

        Returns:
        Predicted labels: 1 for positive class, 0 for negative class.
        """
        # Compute dot product of input features and parameter vector
        pred_num = np.dot(X, self.param)
        
        # Apply threshold to classify predictions as 1 or 0
        pred = np.where(pred_num > 0, 1, 0)
        
        return pred