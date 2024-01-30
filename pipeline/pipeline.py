from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.model_selection import GridSearchCV

def grid_search_models(models, param_grids, X_train, y_train, cv=5):
    best_models = []

    for model, param_grid in zip(models, param_grids):
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_models.append(best_model)
    
    return best_models



# Perform grid search for each model
best_models = grid_search_models(models, param_grids, X_train, y_train)

# Now best_models contains the best models after grid search


def one_dim_x_train(
        X, 
        y,
        models, # SVC() kind things, 
        test_size: float, 
        random_state = None,
    ):
    # test train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state)
    acc_list, auc_list, cm_list = [],[],[]
    
    for model in models:
        # Training Model
        model.fit(X_train, y_train)

        # Eval
        y_pred = model.predict(X_test)

        # Computing stats
        fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred)

        acc_list.append(metrics.accuracy_score(y_test, y_pred))
        auc_list.append(round(metrics.auc(fpr, tpr), 2))
        cm_list.append(confusion_matrix(y_test, y_pred))
        
    return acc_list, auc_list, cm_list
