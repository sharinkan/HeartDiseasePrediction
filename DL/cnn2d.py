import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

class CNN2D(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape, num_classes, learning_rate=0.001, epochs=10, batch_size=32):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X, y):
        self.model = self.build_model()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict_proba(self, X):
        return self.model.predict(X)

    def predict(self, X):
        proba = self.model.predict(X)
        results = [list(p).index(max(p)) for p in proba]
        return np.array(results)

    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]