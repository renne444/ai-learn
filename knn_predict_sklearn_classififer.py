from sklearn.neighbors import KNeighborsClassifier
from metrics import accuracy as my_ac
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()
X = iris_dataset.data
y = iris_dataset.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

param_gird = [
    {
        'weights': ['uniform', 'distance'],
        'n_neighbors': [i for i in range(1, 30)]
    }
]

from sklearn.model_selection import GridSearchCV


kNN_Classifier = KNeighborsClassifier()
grid_search = GridSearchCV(kNN_Classifier, param_gird)

model = grid_search.fit(X, y)
print(model)

best_params = grid_search.best_estimator_
print(best_params)
