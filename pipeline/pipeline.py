from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def one_dim_x_train(
        X, 
        y,
        models, # SVC() kind things, 
        test_size: float, 
        random_state = None,
    ):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state)
    
    acc_list, auc_list, cm_list = [],[],[]
    
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred)
        
        acc_list.append(metrics.accuracy_score(y_test, y_pred))
        auc_list.append(round(metrics.auc(fpr, tpr), 2))
        cm_list.append(confusion_matrix(y_test, y_pred))
        
    return acc_list, auc_list, cm_list
