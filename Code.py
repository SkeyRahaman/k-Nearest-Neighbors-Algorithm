import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from KNearestNeighbors import KNearestNeighbours
from sklearn.metrics import accuracy_score


data = sns.load_dataset("iris")
x = data.iloc[:, 0:4].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25)
knn = KNearestNeighbours(n_neighbors=3, weights="distance")
knn.fit(x_train=x_train, y_train=y_train)

y_pred = knn.predict(np.array(x_test).reshape(len(x_test), len(x_test[0])))

print("Accuracy:- " , round((accuracy_score(y_test,y_pred) * 100), 2), "%")

knn = KNearestNeighbours(n_neighbors=3)
knn.fit(x_train=x_train, y_train=y_train)

y_pred = knn.predict(np.array(x_test).reshape(len(x_test), len(x_test[0])))

print("Accuracy:- " , round((accuracy_score(y_test,y_pred) * 100), 2), "%")



