import operator
from collections import Counter


class KNearestNeighbours:
    def __init__(self, n_neighbors, weights='uniform'):
        self.n_neighbors = n_neighbors  # n_neighbors is the number of n n_neighbors with which the comparison will take plae.
        self.result = []                # result store the overall result result of this algorithm.
        self.x_train = []               # x_trane and y_trane is the tranning data.
        self.y_train = []
        self.weight = 1
        self.set_weights(weights)

    def set_weights(self, weights):
        if weights == "uniform":
            self.weight = -1
        elif weights == "distance":
            self.weight = -1
        else:
            exit(print("Invalid weight value"))



    def fit(self, x_train, y_train):
        self.x_train = x_train          # Stores the tranning data.
        self.y_train = y_train          # Stores the tranning result.
        print("Tranning done!")

    def predict(self, x_test):      # x_test is the testing data.
        for j in x_test:            # Loops around each value in x_test or number of values the algorithm need to predict.
            distance = {}           # Distance stores all the distance between each point in x_traie and the test point
            counter = 0             # keep a track of index number of x_trane to extract the respective result from y_trane
            for i in self.x_train:  # Loops around each value in x_trane or number of diamention this algorithm will use.
                sum_of_the_square = 0   # Stores the incomplete distance between two point.
                for k in range(len(j)):
                    sum_of_the_square += (j[k] - i[k]) ** 2
                distance[counter] = sum_of_the_square ** self.weight / 2
                counter += 1
            distance = sorted(distance.items(), key=operator.itemgetter(1))
            self.result.append(self.classify(distance[:self.n_neighbors]))    # Classified result is stored in required variable which will be returned to the user.
            del distance
        return self.result

    def classify(self, distance):   # Derive the result of this algorithm.
        label = []
        for i in distance:
            label.append(self.y_train[i[0]])
        return Counter(label).most_common()[0][0]
