"""showing stats / results
"""

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def view_cm(models : "MLmodel", cm_list:list) -> None:
    """plot confusion matrix

    Args:
        models (MLmodel): models
        cm_list (list): confusion matrix 
    """
    model_names = [model.__class__.__name__ for model in models]
    
    fig = plt.figure(figsize = (18, 10))
    for i in range(len(cm_list)):
        cm = cm_list[i]
        model = model_names[i]
        sub = fig.add_subplot(2, 3, i+1).set_title(model)
        cm_plot = sns.heatmap(cm, annot=True, cmap = 'Blues_r')
        cm_plot.set_xlabel('Predicted Values')
        cm_plot.set_ylabel('Actual Values')
        
    plt.show()
        
def get_acc_auc_df(models : "MLmodel", acc_list : list, auc_list : list) ->pd.DataFrame:
    """accuarcy and AUC dataframe

    Args:
        models (MLmodel): models
        acc_list (list): accuaray list
        auc_list (list): AUC list

    Returns:
        pd.DataFrame: containing both list
    """
    model_names = [model.__class__.__name__ if not(hasattr(model, "feature_train_name")) else model.feature_train_name for model in models]
    return pd.DataFrame({'Model': model_names, 'Accuracy': acc_list, 'AUC': auc_list})



def show_outcome_distrib(df: pd.DataFrame) -> None:
    """display outcome disturbtion

    Args:
        df (pd.DataFrame): disturbtion
    """
    count = ""
    if isinstance(df, pd.DataFrame):
        count = df['Outcome'].value_counts()
    else:
        count = df.value_counts()

    count.plot(kind = 'pie', explode = [0, 0.1], figsize = (6,6), autopct = '%1.1f%%', shadow = True)

    plt.ylabel("OUtcome: Normal vs Abnormal")
    plt.legend(['Normal', 'Abnormal'])
    plt.show()