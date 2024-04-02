from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,AveragePooling1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

models = []

models.append(LogisticRegression(solver='liblinear'))
models.append(SVC(max_iter=10000))
models.append(KNeighborsClassifier())
models.append(DecisionTreeClassifier())
models.append(RandomForestClassifier())
models.append(GaussianNB())

param_grids = [
    {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}, # Logistic Regression
    {'C': [1, 10, 100, 1000], 'kernel': ['linear', 'rbf'], 'gamma': [0.1, 0.01, 0.001]}, # SVC
    {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], 'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'], 'weights': ['uniform', 'distance']}, # KNeighborsClassifier
    {'max_depth': [None, 5, 10, 15, 20, 25, 30], 'min_samples_split': [2, 5, 10, 15, 20], 'min_samples_leaf': [1, 2, 4, 6, 8]},# DecisionTreeClassifier
    {'criterion': ['gini', 'entropy'], 'n_estimators': [100, 150, 200, 250, 300]},# RandomForestClassifier
    {}  # GaussianNB doesn't have hyperparameters
]

def get_cnn_model(input_shape):
    cnn_model = Sequential()

    # Convolutional layer
    cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Dropout(0.2))

    # Another convolutional layer
    cnn_model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Dropout(0.2))

    # Third convolutional layer
    cnn_model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Dropout(0.3))

    # Flattening followed by dense layers
    cnn_model.add(Flatten())
    cnn_model.add(Dense(256, activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(1, activation='sigmoid'))  
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    cnn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return cnn_model
