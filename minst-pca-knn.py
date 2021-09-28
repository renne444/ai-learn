from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import scipy.io as scio

mnist = scio.loadmat('D:\\data\\develop\\ml\\ai-learn\\mnist-original.mat')
mnist_data = np.array(mnist['data']).transpose()
mnist_label = np.array(mnist['label']).transpose()

X_train, X_test, y_train, y_test \
    = train_test_split(mnist_data, mnist_label, test_size=0.2, random_state=666, shuffle=True)

print(X_train.shape)
print(y_train[0])
plt.imshow(X_train[0].reshape(28, 28))
plt.show()

from sklearn.decomposition import PCA

pca = PCA(0.9)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_train_zip = pca.inverse_transform(X_train_reduction)
plt.imshow(X_train_zip[0].reshape(28, 28))
plt.show()

from sklearn.neighbors import KNeighborsClassifier
