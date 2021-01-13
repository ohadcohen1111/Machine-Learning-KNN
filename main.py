import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from termcolor import colored


def euclidean_distance(a,b):
    sum=0
    for a1,b1 in zip(a,b):
        sum += (a1-b1)**2
    return np.sqrt(sum)

def manhattan_norm(a,b):
    sum=0
    for a1,b1 in zip(a,b):
        sum += np.abs(a1-b1)
    return sum

def frechet_distance(a,b):
    max=0
    for a1,b1 in zip(a,b):
        if np.abs(a1-b1) > max:
            max = np.abs(a1-b1)
    return max

class KNN:

    def __init__(self,n_neighbors):
        self._k = n_neighbors

    def fit(self, X1, y1):
        self._X1 = X1
        self._y1 = y1

    # predict the point (one point)
    def predict_one(self, X12, lp):
        distances = []
        for X11, y11 in zip(self._X1, self._y1):
            if lp == 1:
                dist = manhattan_norm(X11, X12)
            elif lp == 2:
                dist = euclidean_distance(X11, X12)
            elif lp == 3:
                dist = frechet_distance(X11, X12)

            distances.append([y11, dist])
        distances.sort(key=lambda elem: elem[1])
        # array of k nearest neighbors
        kdistances = distances[:self._k]
        klabels = [row[0] for row in kdistances]
        return max(klabels, key=klabels.count)

    # get array of predict test
    def predict(self, XPred, lp):
        return np.array([self.predict_one(x, lp) for x in XPred])

    def score(self, X_test, y_test, lp):
        pred = self.predict(X_test, lp)
        return np.sum(pred==y_test) / y_test.shape[0]





def main():
    for k in range(1, 10, 2):
        print(colored(("------------------------------   k : {}  ------------------------------".format(k)), 'red'))
        for lp in range(1, 4):
            sum_test=0
            sum_train=0
            for i in range(100):
                # split the data
                df = pd.read_fwf('HC_Body_Temperature.txt')
                # get 0-3 columns in jumps of 2
                X = df.iloc[:, 0:3:2].to_numpy()
                y = df.iloc[:, 1].to_numpy()
                # split the data into train and test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

                # initiate the object
                knn2 = KNN(n_neighbors=k)
                knn2.fit(X_train, y_train)
                # pred2 = knn2.predict(X_test, 3)

                # print the statistics
                sum_train += knn2.score(X_train, y_train, lp)
                sum_test += knn2.score(X_test, y_test, lp)

            print("lp : {}  score train: {} score test: {}".format(lp, sum_train/100, sum_test/100))

            
if __name__ == '__main__':
    main()
