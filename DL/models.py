import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Concatenate, GlobalAveragePooling2D, Reshape, Multiply
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


class MultiScaleCNN(CNN2D):
    # This model is an optimized CNN based on the paper (EGG Transformer):
    # https://www.sciencedirect.com/science/article/abs/pii/S1746809423011473
    def build_model(self):
        inputs = Input(shape=self.input_shape)
        
        # First scale: small filter
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # Second scale: medium filter
        conv2 = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # Third scale: large filter
        conv3 = Conv2D(32, (7, 7), activation='relu', padding='same')(inputs)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # Concatenate the feature maps from different scales
        concat = Concatenate()([pool1, pool2, pool3])

        # Apply Average Pooling after concatenation
        avg_pool = AveragePooling2D(pool_size=(2, 2))(concat)

        # Channel Recalibration
        channels = avg_pool.shape[-1]
        x = GlobalAveragePooling2D()(avg_pool)
        x = Reshape((1, 1, channels))(x)
        x = Dense(channels // 16, activation='relu')(x)
        x = Dense(channels, activation='sigmoid')(x)
        recalibrated = Multiply()([avg_pool, x])

        # Further convolutional layers
        conv4 = Conv2D(64, (3, 3), activation='relu')(recalibrated)
        pool4 = MaxPooling2D((2, 2))(conv4)
        conv5 = Conv2D(128, (3, 3), activation='relu')(pool4)
        pool5 = MaxPooling2D((2, 2))(conv5)

        # Flatten and Dense layers
        flatten = Flatten()(pool5)
        dense1 = Dense(128, activation='relu')(flatten)
        outputs = Dense(self.num_classes, activation='softmax')(dense1)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model
    