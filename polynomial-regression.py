import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(-3, 3, size=100)
X = x.reshape(-1, 1)

y = 0.5 * x**2 + x + x + np.random.normal(0, 1, size=100)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_predict1 = lin_reg.predict(X)


X2 = np.hstack([X, X**2])
lin_reg2 = LinearRegression()
lin_reg2.fit(X2, y)
y_predict2 = lin_reg2.predict(X2)

plt.scatter(x, y, color="blue")
plt.plot(x, y_predict1, color="red")
plt.plot(np.sort(x), y_predict2[np.argsort(x)], color="orange")
plt.show()

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Lasso
pipe_lasso_regression = Pipeline([
    ("poly", PolynomialFeatures(degree=50)),
    ("std_scaler", StandardScaler()),
    ("lasso_reg", Lasso(alpha=0.01)),
])
pipe_lasso_regression.fit(X)
y_lasso = pipe_lasso_regression.predict(X)

