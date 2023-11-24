# Should use cli options to run different things

from pipeline.models import models
from pipeline.pipeline import one_dim_x_train
from pipeline.stats import view_cm, get_acc_auc_df, show_outcome_distrib
from pipeline.preprocessing import * # fix later

from tqdm import tqdm

def pipeline(original_data):
    model_df = data_wrangling(original_data)
    
    X = one_hot_encoding(model_df, ['Sex', 'Murmur', 'Age', 'Systolic murmur quality', 'Systolic murmur pitch',
                                'Systolic murmur grading', 'Systolic murmur shape', 'Systolic murmur timing', 'Most audible location'
                               ])
    y = model_df['Outcome']
    
    # To verify if there's potential need for balancing the dataset
    show_outcome_distrib(y) 
    acc_list, auc_list, cm_list = one_dim_x_train(X, y, models=models,test_size=0.2, random_state=0)
    view_cm(models, cm_list)
    
    my_df = get_acc_auc_df(models, acc_list, auc_list)
    print(my_df)
    
    
if __name__ == "__main__":
    from pathlib import Path
    file = Path(".") / "assets" / "the-circor-digiscope-phonocardiogram-dataset-1.0.3"
    original_data = pd.read_csv(str(file  / "training_data.csv"))
    pipeline(original_data)