# Import necessary packages
import pandas as pd
import numpy as np
from LogisticRegression import LogisticRegression  # Custom Logistic Regression class
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("***\\logistic_regression\\creditcard.csv")

# Add a bias column to the dataset
data["bias"] = 1

# Drop the 'Time' column, as it's not needed for classification
data = data.drop(columns=['Time'])

# Separate features and target variable
Y = data['Class'].tolist()
data = data.drop(columns=['Class'])
X = data

# Define hyperparameters for tuning
test_size = 0.2        # Proportion of data for testing
val_size = 0.2         # Proportion of data for validation
range_learning_rate = [1, 30]  # Range of learning rates to test
lern_trys = 10                     # Number of learning rates to test within the range
range_max_iter = [1, 10]    # Range for max iterations to test
max_iter_trys = 10                 # Number of max iterations to test within the range
err = 1e-3                     # Error tolerance for gradient descent

# Split the dataset into training/validation and test data
X_train_val, X_test, y_train_val, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

# Further split training/validation data into separate training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size/(1-test_size), random_state=42)

# Initialize storage for model accuracy scores and trained models
all_accuracy_scores = []
all_models = []

# Set up learning rates and max iteration values for hyperparameter tuning
learning_rates = [round(range_learning_rate[0] + i * (range_learning_rate[1] - range_learning_rate[0]) / (lern_trys - 1)) for i in range(lern_trys)]
max_iters = [round(range_max_iter[0] + i * (range_max_iter[1] - range_max_iter[0]) / (max_iter_trys - 1)) for i in range(max_iter_trys)]

# Loop through each combination of learning rate and max iteration values
for learning_rate in learning_rates:
    max_iters_models = []
    max_iters_accuracy = []
    
    for max_iter in max_iters:
        # Train a logistic regression model with the current hyperparameters
        log_reg_clf = LogisticRegression(learning_rate, max_iter, err=err)
        log_reg_clf.fit(X_train, y_train)
        
        # Predict on the validation set
        y_pred = log_reg_clf.predict(X_val)
        
        # Compute and store accuracy
        accuracy = accuracy_score(y_val, y_pred)
        max_iters_accuracy.append(accuracy)
        max_iters_models.append(log_reg_clf)
    
    # Store models and accuracy scores for this learning rate
    all_models.append(max_iters_models)
    all_accuracy_scores.append(max_iters_accuracy)

# Find the best model based on accuracy
index_row_lern, index_col_lern_iter = np.unravel_index(np.argmax(all_accuracy_scores), np.array(all_accuracy_scores).shape)
best_log_reg_clf = all_models[index_row_lern][index_col_lern_iter]
best_learning_rate = learning_rates[index_row_lern]
best_max_iters = max_iters[index_col_lern_iter]

# Display best hyperparameters found
print("Best learning rate: ", best_learning_rate)
print("Best max iterations: ", best_max_iters)

# Evaluate the best model on the test set
y_pred = best_log_reg_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy: ", accuracy)