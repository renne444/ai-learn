from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

class LinearModel:

    def __init__(self, feature_num, learn_rate):
        self.w = np.full(feature_num + 1, 0, dtype=float)
        self.learn_rate = learn_rate
        self.batch_size = 5
        pass

    def expand(self, X):
        X_append = np.full(X.shape[0], 1).reshape(1, -1)
        X = np.concatenate((X, X_append.transpose()), axis=1)
        return X

    def fit_batch(self, X, y):
        a = np.full(1,0)
        y_pred = X.dot(self.w)
        diff_mean = np.mean(y_pred - y)
        X_mean = np.mean(X, axis=0)

        w_grad = X_mean * self.learn_rate * diff_mean
        self.w = self.w - w_grad

    def fit(self, X, y):
        X = self.expand(X)
        batch_size = self.batch_size
        for i in range(int(X.shape[0] / batch_size)):
            X_train = X[i * batch_size: (i + 1) * batch_size]
            y_train = y[i * batch_size: (i + 1) * batch_size]
            self.fit_batch(X_train, y_train)

        _, train_loss = self.test(X, y)
        return train_loss

    def test(self, X, y):
        y_pred = X.dot(self.w)
        std_loss = np.sqrt(np.mean((y_pred - y) ** 2))
        return y_pred, std_loss

    def predict(self, X, y):
        X = self.expand(X)
        return self.test(X, y)

    def normalization(self, X, y, test_flag=False):
        if not test_flag:
            self.X_mean = X_mean = np.mean(X, axis=0)
            self.X_std = X_std = np.std(X, axis=0)
            self.y_mean = y_mean = np.mean(y, axis=0)
            self.y_std = y_std = np.std(y, axis=0)
        else:
            X_mean = self.X_mean
            X_std = self.X_std
            y_mean = self.y_mean
            y_std = self.y_std


        X = (X - X_mean) / X_std
        y = (y - y_mean) / y_std

        return X, y


loaded_data = datasets.load_boston()
X_data = loaded_data.data
X_data = X_data[:, [0, 1, 3, 4, 5, 7, 8, 9, 10, 11, 12]]

y_data = loaded_data.target

model = LinearModel(X_data.shape[1], 0.002)
X_data, y_data = model.normalization(X_data, y_data)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0, shuffle=True)
X_train, y_train = model.normalization(X_train, y_train)
X_test, y_test = model.normalization(X_test, y_test, test_flag=True)

epoch = 2000
training_loss_array = []
testing_loss_array = []
w_array = []
for i in range(epoch):
    shuffle_array = np.random.permutation(X_train.shape[0])
    X_train = X_train[shuffle_array]
    y_train = y_train[shuffle_array]

    train_loss = model.fit(X_train, y_train)
    y_pred, test_loss = model.predict(X_test, y_test)

    training_loss_array = training_loss_array + [train_loss]
    testing_loss_array = testing_loss_array + [test_loss]
    w_array = w_array + [model.w]

np_training_loss = np.array(training_loss_array)
np_testing_loss = np.array(testing_loss_array)
print("training loss", np_training_loss[-1])
print("testing loss", np_testing_loss[-1])
print("min training loss", np.min(np_training_loss))
print("min testing loss", np.min(np_testing_loss))

import matplotlib.pyplot as plt

plt.subplot(2, 1, 1)
plt.plot(np_training_loss)
plt.plot(np_testing_loss)
plt.title("loss")

plt.subplot(2, 1, 2)
np_w_array = np.array(w_array)
np_w_array = np_w_array.transpose()
for i in range(np_w_array.shape[0]):
    plt.plot(np_w_array[i])
plt.title("w")
plt.savefig("boston.jpg")
plt.show()

print(model.w)
print(np.argsort(np.abs(model.w)))
