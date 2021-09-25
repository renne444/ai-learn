import numpy as np
import matplotlib.pyplot as plt
from util import demean, direction

data = np.empty((100, 2))
data[:, 0] = np.random.uniform(0., 100., size=100)
data[:, 1] = 0.75 * data[:, 0] + 3 + np.random.normal(0, 10., size=100)


class PCA:
    def __init__(self, dim):
        self.lr = 0.01
        self.w = np.full(dim, 1., dtype=float)
        self.history_var = []

    def mapping_standard_deviation(self, X):
        dist = X.dot(self.w.transpose()) ** 2
        sqr = np.sqrt(dist.sum() / X.shape[0])
        return sqr

    def grad_calculate(self, X):
        grad = X.transpose().dot(X.dot(self.w)) * 2. / X.shape[0]
        return grad

    def get_pca(self, X):
        dem_X = demean(X)
        self.w = np.full((dem_X.shape[1]), 1., dtype=float)
        self.w = direction(self.w)
        self.history_var = []

        epoch = 20
        for _ in range(epoch):
            loss = self.mapping_standard_deviation(dem_X)
            w_grad = self.grad_calculate(dem_X)
            self.w += w_grad * self.lr
            self.w = direction(self.w)
            self.history_var.append(loss)


pca = PCA(data.shape[1])
pca.get_pca(data)
plt.title('single PCA')
plt.scatter(data[:, 0], data[:, 1])
plt.plot([0, pca.w[0] * 120], [0, pca.w[1] * 120], color='red')
plt.savefig('./single PCA.jpg')
plt.show()

plt.title('standard deviation for single PCA')
plt.xlabel("epoch")
plt.xlim([0, 20])
plt.xticks(np.arange(20))
plt.ylabel("stand deviation")
plt.plot(np.arange(len(pca.history_var)), pca.history_var)
plt.savefig('./single PCA standard deviation.jpg')
plt.show()
