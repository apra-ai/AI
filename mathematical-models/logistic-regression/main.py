# Import necessary packages
import pandas as pd
import numpy as np
from LogisticRegression import LogisticRegression  # Import custom Logistic Regression implementation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("***\\logistic_regression\\creditcard.csv")

# Add a bias column with a constant value of 1
data["bias"] = 1

# Drop the 'Time' column as it may not be relevant for the classification
data = data.drop(columns=['Time'])

# Separate the target variable 'Class' (which indicates fraud or non-fraud) from the features
Y = data['Class'].tolist()  # Convert target variable to list for compatibility with custom classifier
data = data.drop(columns=['Class'])

# Assign the features to X for clarity
X = data

# Define hyperparameter for train-test split
test_size = 0.2  # 20% of data will be used as test data

# Split the data into training and testing sets, using a fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

# Initialize the custom Logistic Regression classifier with specified hyperparameters
log_reg_clf = LogisticRegression(learning_rate=10, max_iter=500, err=1e-3)

# Train the model on the training data using gradient descent
log_reg_clf.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = log_reg_clf.predict(X_test)

# Calculate and print the accuracy score of the model on the test set
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)