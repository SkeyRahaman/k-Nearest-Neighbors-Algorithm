import operator
from collections import Counter
import numpy as np


class KNearestNeighbours:
    def __init__(self, n_neighbors, weights='uniform'):
        self.n_neighbors = n_neighbors  # n_neighbors is the number of n n_neighbors with which the comparison will take plae.
        self.result = []                # result store the overall result result of this algorithm.
        self.x_train = []               # x_trane and y_trane is the tranning data.
        self.y_train = []
        self.weight = weights

    def fit(self, x_train, y_train):
        self.x_train = x_train          # Stores the tranning data.
        self.y_train = y_train          # Stores the tranning result.
        print("Tranning done!")

    def predict(self, x_test):      # x_test is the testing data.

        for j in x_test:            # Loops around each value in x_test or number of values the algorithm need to predict.

            distance = {}           # Distance stores all the distance between each point in x_traie and the test point
            counter = 0             # keep a track of index number of x_trane to extract the respective result from y_trane
            for i in self.x_train:  # Loops around each value in x_trane or number of dimension this algorithm will use.
                sum_of_the_square = 0   # Stores the incomplete distance between two point.
                if len(j) == 1:
                    distance[counter] = abs(j - i)

                else:
                    for k in range(len(j)):
                        sum_of_the_square += (j[k] - i[k]) ** 2
                    distance[counter] = sum_of_the_square ** 1 / 2
                counter += 1

            distance = sorted(distance.items(), key=operator.itemgetter(1))

            self.result.append(self.classify(distance[:self.n_neighbors]))    # Classified result is stored in required variable which will be returned to the user.
            del distance
        if len(self.result) == 1:
            return self.result[0]
        else:
            return self.result

    def classify(self, distance):   # Derive the result of this algorithm.
        distance = dict(distance)
        if self.weight == "distance":
            weighted_prediction = {}
            for i in distance:
                if distance[i] != 0:
                    distance[i] **= -1
            number_of_unique_entry_in_y_test = np.unique(self.y_train)
            for i in number_of_unique_entry_in_y_test:
                weighted_prediction[i] = 0
                for j in distance:
                    if self.y_train[j] == i:
                        weighted_prediction[i] += distance[j]
            return sorted(weighted_prediction.items(), key=operator.itemgetter(1), reverse=True)[0][0]

        else:
            label = []
            for i in distance:
                label.append(self.y_train[i])
            return Counter(label).most_common()[0][0]



class KNearestRegressor(KNearestNeighbours):
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors  # n_neighbors is the number of n n_neighbors with which the comparison will take plae.
        self.result = []                # result store the overall result result of this algorithm.
        self.x_train = []               # x_trane and y_trane is the tranning data.
        self.y_train = []

    def classify(self, distance):   # Derive the result of this algorithm.
        label = []
        for i in distance:
            label.append(self.y_train[i[0]])
        return sum(label)/len(label)
