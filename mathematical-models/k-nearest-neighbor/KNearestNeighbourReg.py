import numpy as np

class KNearestNeighborReg():
    def __init__(self,k=5):
        self.K = k

    def fit(self,X_train,Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    #Euclidean distance used for Distance
    def predict(self, X_test):
        # Compute all distances at once using broadcasting
        distances = np.sqrt(((X_test[:, np.newaxis] - self.X_train) ** 2).sum(axis=2))

        # Get the indices of the K nearest neighbors for each test sample
        K_nearest_indices = np.argsort(distances, axis=1)[:, :self.K]

        # Get the labels of the K nearest neighbors
        y_values_k_nearest = self.Y_train[K_nearest_indices]

        # For each test sample, find the most common label
        y_pred = np.mean(y_values_k_nearest,axis=1)
        return y_pred