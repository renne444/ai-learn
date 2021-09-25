import numpy as np
import matplotlib.pyplot as plt

data = np.empty((100, 2))
data[:, 0] = np.random.uniform(0., 100., size=100)
data[:, 1] = 0.75 * data[:, 0] + 3 + np.random.normal(0, 10., size=100)

lr = 0.01


def demean(X):
    return X - np.mean(X)

def direction(w):
    return w / np.linalg.norm(w)


def mapping_standard_deviation(X, w):
    dist = X.dot(w.transpose()) ** 2
    sqr = np.sqrt(dist.sum() / X.shape[0])
    return sqr

def grad_calculate(X, w):
    grad = X.transpose().dot(X.dot(w)) * 2. / X.shape[0]
    return grad


def get_pca(X):
    dem_X = demean(X)
    w = np.full(2, 1., dtype=float)
    w = direction(w)
    history_var = []

    epoch = 20
    for _ in range(epoch):
        loss = mapping_standard_deviation(dem_X, w)
        w_grad = grad_calculate(dem_X, w)
        w += w_grad * lr
        w = direction(w)
        history_var.append(loss)
    return w, history_var


pca, history = get_pca(data)
plt.title('single PCA')
plt.scatter(data[:, 0], data[:, 1])
plt.plot([0, pca[0] * 120], [0, pca[1] * 120], color='red')
plt.savefig('./single PCA.jpg')
plt.show()

plt.title('standard deviation for single PCA')
plt.xlabel("epoch")
plt.xlim([0, 20])
plt.xticks(np.arange(20))
plt.ylabel("stand deviation")
plt.plot(np.arange(len(history)), history)
plt.savefig('./single PCA standard deviation.jpg')
plt.show()
