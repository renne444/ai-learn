from sklearn.neighbors import KNeighborsClassifier
from metrics import accuracy as my_ac
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

iris_dataset = load_iris()
X = iris_dataset.data
y = iris_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666, shuffle=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_scale = scaler.transform(X_train)

class MyStandScale:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X_train):
        assert X_train.ndim == 2
        self.mean_ = np.mean(X_train, axis=0)
        self.scale_ = np.std(X_train, axis=0)

    def transform(self, X):
        X_float = np.array(X, dtype=float)
        return (X_float - self.mean_) / self.scale_

scaler2 = MyStandScale()
scaler2.fit(X_train)
res2 = scaler2.transform(X_train)
assert np.all(res2 == X_scale)
