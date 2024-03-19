from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.model_selection import GridSearchCV
from pipeline.models import get_cnn_model

from sklearn.metrics import confusion_matrix
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score

def grid_search_models(models, param_grids, X_train, y_train, cv=5):
    best_models = []

    for model, param_grid in zip(models, param_grids):
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_models.append(best_model)
    
    return best_models

def one_dim_x_train(
        X, 
        y,
        models, # SVC() kind things, 
        param_grids,
        test_size: float, 
        random_state = None,
    ):
    # test train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state)
    acc_list, auc_list, cm_list = [],[],[]

    best_models = grid_search_models(models, param_grids, X_train, y_train)
        
    for model in best_models:
        # Training Model
        model.fit(X_train, y_train)

        # Eval
        y_pred = model.predict(X_test)

        # Computing stats
        fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred)

        acc_list.append(metrics.accuracy_score(y_test, y_pred))
        auc_list.append(round(metrics.auc(fpr, tpr), 2))
        cm_list.append(confusion_matrix(y_test, y_pred))
    print(best_models)
    print(acc_list)
    print(auc_list)
        
    return acc_list, auc_list, cm_list

def cnn_train(X,y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    cnn = get_cnn_model((X_train.shape[1],1))
    cnn.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_val, y_val), verbose=1)

    probabilities = cnn.predict(X_test)
    threshold = 0.5
    y_pred = (probabilities >= threshold).astype(int)

    # y_pred = np.round(y_pred).astype(int)  # Convert probabilities to binary labels

    acc = metrics.accuracy_score(y_test, y_pred)
    fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc}")
    print(f"Auc: {auc}")
    print(f"F1 Score: {f1}")
    acc = round(acc * 100, 2)
    auc = round(auc * 100, 2)
    f1 = round(f1 * 100, 2)
    return acc, auc, f1


