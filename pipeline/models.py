from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,Lambda, AveragePooling1D, MaxPooling1D, Flatten,Reshape, Dense, Dropout, BatchNormalization, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

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

def get_cnn_with_concat(input_shapes):
    print(input_shapes)

    cnn_list = []
    input_list = []
    max_seq_length = max(input_shapes) 

    for i, input_shape in enumerate(input_shapes):
        input = tf.keras.Input(shape=(input_shape,1))
        cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(input)
        cnn = Reshape((-1,64))(cnn) 
        
        padding_shape = tf.constant([[0, 0], [0, max_seq_length - input_shape], [0, 0]])
        cnn = Lambda(lambda x, padding_shape=padding_shape: tf.pad(x, padding_shape, 'CONSTANT'))(cnn)

        # cnn = Sequential()
        # cnn.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(shape,1)))
        cnn_list.append(cnn)
        input_list.append(input)

    combined_features = concatenate(cnn_list, axis=-1)

    # # Pass concatenated features to another CNN
    # x = Conv1D(filters=128, kernel_size=3, activation='relu')(combined_features)
    # x = MaxPooling1D(pool_size=2)(x)
    # x = Flatten()(x)
    # x = Dense(256, activation='relu')(x)
    # output = Dense(1, activation='sigmoid')(x)

    # # Create the final model
    # model = tf.keras.Model(inputs=input_list, outputs=output)
    # model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # CNN architecture
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(combined_features)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # x = Conv1D(filters=256, kernel_size=3, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling1D(pool_size=2)(x)
    # x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_list, outputs=output)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model 

    # cnn_model = Sequential()

    # # Convolutional layer
    # cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    # cnn_model.add(BatchNormalization())
    # cnn_model.add(MaxPooling1D(pool_size=2))
    # cnn_model.add(Dropout(0.2))

    # # Another convolutional layer
    # cnn_model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    # cnn_model.add(BatchNormalization())
    # cnn_model.add(MaxPooling1D(pool_size=2))
    # cnn_model.add(Dropout(0.2))

    # # Third convolutional layer
    # cnn_model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    # cnn_model.add(BatchNormalization())
    # cnn_model.add(MaxPooling1D(pool_size=2))
    # cnn_model.add(Dropout(0.3))

    # # Flattening followed by dense layers
    # cnn_model.add(Flatten())
    # cnn_model.add(Dense(256, activation='relu'))
    # cnn_model.add(Dropout(0.5))
    # cnn_model.add(Dense(1, activation='sigmoid'))  
    # # Compile the model
    # optimizer = Adam(learning_rate=0.001)
    # cnn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # return cnn_model
