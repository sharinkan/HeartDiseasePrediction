"""Models and Parameters loading
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Adding models
models = []
models.append(LogisticRegression(solver='liblinear'))
models.append(SVC())
models.append(KNeighborsClassifier(n_neighbors=7))
models.append(DecisionTreeClassifier())
models.append(RandomForestClassifier())
models.append(GaussianNB())

# adding param grid and ensuring order match
param_grid_dict = {
    "LogisticRegression" : {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']},
    "SVC" : {'C': [1, 10, 100, 1000], 'kernel': ['linear', 'rbf'], 'gamma': [0.1, 0.01, 0.001]},
    "KNeighborsClassifier" : {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], 'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], 'weights': ['uniform', 'distance']},
    "DecisionTreeClassifier" : {'max_depth': [None, 5, 10, 15, 20, 25, 30], 'min_samples_split': [2, 5, 10, 15, 20], 'min_samples_leaf': [1, 2, 4, 6, 8]},
    "RandomForestClassifier" : {'criterion': ['gini', 'entropy'], 'n_estimators': [100, 150, 200, 250, 300]},
    "GaussianNB" : {} # GaussianNB doesn't have hyperparameters
}

param_grids = [
    param_grid_dict[model.__class__.__name__] for model in models
]