import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from KNearestNeighbourReg import KNearestNeighborReg

dicetory = "***\\k-nearest_neighbors\\kc_house_data_small_train.csv"
data = pd.read_csv(dicetory)

Y = data["price"]
X = data.drop(columns=["price","id","date"])

#Normalise x-features
X_values = X.to_numpy()
for i, col in enumerate(X):
    col_min = X[col].min()
    col_max = X[col].max()
    if col_min==col_max:
        X_values[i] = X_values[i]*0
    else:
        X_values[i] = ((X_values[i]-col_min)/(col_max-col_min))*2-1

#Split into train, test, val Data
X_train_val, X_test, y_train_val, y_test = train_test_split(X_values, Y.to_numpy(), test_size=0.2, random_state=11)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=11)

#predict the best k for KNN
test_k=[1,2,3,4,5,6,7,8]
predictions_rmse = []
models=[]
for k in test_k:
    k_near_reg = KNearestNeighborReg(k)
    k_near_reg.fit(X_train,y_train)
    y_pred = k_near_reg.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val,y_pred))
    predictions_rmse.append(rmse)
    models.append(k_near_reg)

#final predict of rmse with test data
y_pred_test = models[predictions_rmse.index(min(predictions_rmse))].predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test,y_pred_test))
print(rmse_test)