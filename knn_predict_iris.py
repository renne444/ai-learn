import numpy as np
from knn import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
X = iris_dataset.data
y = iris_dataset.target

# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

# %%

import datetime

knn = KNN()
knn.train(X_train, y_train)
rate_x = np.arange(1, 35)
rate = []
time_cost = []
for k in rate_x:
    st = datetime.datetime.now().microsecond / 1000
    y_pred = knn.predict(X_test, k)
    correct_rate = np.sum(y_pred == y_test) / y_pred.shape[0] * 100
    et = datetime.datetime.now().microsecond / 1000
    rate.append(correct_rate)
    time_cost.append(et - st)

import matplotlib.pyplot as plt

plt.ylim(90, 103)
plt.xlabel("K")
plt.ylabel("correct rate %")
plt.title("iris dataset - KNN classification - correct rate")
plt.plot(rate_x, rate)
plt.savefig("./KNN correct rate.jpg")
plt.show()

plt.title("iris dataset - KNN classification - time cost")
plt.xlabel("K")
plt.ylabel("time cost ms")
plt.plot(rate_x, time_cost)
plt.savefig("./KNN time cost.jpg")
plt.show()


