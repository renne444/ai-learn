import numpy as np

class KNN:
    def __init__(self):
        pass

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # 计算每个测试样本到所有训练样本的距离
    # 结果格式：dist[测试样本下标][训练样本下标] -> 欧几里得距离
    def dist_to_train(self, X_test):
        train_sample_size = self.y_train.shape[0]
        test_sample_size = X_test.shape[0]
        dim = self.X_train.shape[1]

        A = self.X_train.reshape(1, -1, dim).repeat(test_sample_size, axis=0)
        B = X_test.repeat(train_sample_size, axis=0).reshape(test_sample_size, train_sample_size, dim)
        D = A - B

        dist = np.sqrt(np.sum(D ** 2, axis=2))
        return dist

    # 计算距离最近的K个样本的结果，并取数量最多的结果
    def predict(self, X_test, k):
        dist = self.dist_to_train(X_test)

        ns = np.argsort(dist, axis=1)[:, :k]
        nr = self.y_train[ns]
        C = np.array([np.sum(nr == i, axis=1) for i in range(3)]).transpose()
        C = np.argmax(C, axis=1)

        return C
