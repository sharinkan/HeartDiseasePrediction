from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB



models = []

models.append(LogisticRegression(solver='liblinear'))
models.append(SVC())
models.append(KNeighborsClassifier(n_neighbors=7))
models.append(DecisionTreeClassifier())
models.append(RandomForestClassifier())
models.append(GaussianNB())


if __name__ == "__main__":
    test_model = SVC()