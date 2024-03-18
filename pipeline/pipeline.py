from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def one_dim_x_train(
        X, 
        y,
        models, # SVC() kind things, 
        test_size: float, 
        random_state = None,
    ):
    # test train split

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = test_size, random_state=random_state)
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
        
    return acc_list, auc_list, cm_list, (X_test, y_test)

def mixture_one_dim_x_train(
        Xs : dict,  # feat_name : feat_ary
        ys : dict, # feat_name : feat_ary
        models_feat : dict, # model : feat_name
        test_size: float, 
        random_state = 1, # make this fixed, or it might not work too well.. as it might give some model trained data as testing data
    ):
    # test train split

    acc_list, auc_list, cm_list = [],[],[]
    test_Xs, test_ys = {}, {}
    for model, feat_name in models_feat.items():
        X_train, X_test, y_train, y_test = train_test_split(Xs[feat_name], ys[feat_name], test_size = test_size, random_state=random_state)
        test_Xs[feat_name] = X_test
        test_ys[feat_name] = y_test

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred)

        acc_list.append(metrics.accuracy_score(y_test, y_pred))
        auc_list.append(round(metrics.auc(fpr, tpr), 2))
        cm_list.append(confusion_matrix(y_test, y_pred))

    return acc_list, auc_list, cm_list, (test_Xs, test_ys)