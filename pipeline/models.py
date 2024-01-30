from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB



models = []

models.append(LogisticRegression(solver='liblinear'))
models.append(SVC())
models.append(KNeighborsClassifier())
models.append(DecisionTreeClassifier())
models.append(RandomForestClassifier())
models.append(GaussianNB())

param_grids = [
    {'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']},
    {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    {'n_neighbors': [3, 5, 7, 9]},
    {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
    {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30]},
    {},  # GaussianNB doesn't have hyperparameters
]
