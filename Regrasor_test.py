import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from KNearestNeighbors import KNearestRegressor
from  sklearn.metrics import r2_score


data = pd.read_csv("Salary_Data.csv")
x = data.iloc[:, 0].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25)
knn = KNearestRegressor(n_neighbors=3)
knn.fit(x_train=x_train,y_train=y_train)

y_pred = knn.predict(np.array(x_test).reshape(len(x_test), 1))



print("Accuracy:- " , round((r2_score(y_test, y_pred) * 100), 2), "%")




