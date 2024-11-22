import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from FeedforwardNeuralNetwork import FeedforwardNeuralNetwork
import numpy as np

directory = "***\\Regression\\FNN\\WineQT.csv"
data = pd.read_csv(directory)

Y = data["quality"]
X = data.drop(columns=["quality", "Id"])

X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

fnn = FeedforwardNeuralNetwork()
fnn.fit(X_train,y_train)

y_pred = fnn.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,y_pred.T))
print(rmse)