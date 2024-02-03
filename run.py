# Should use cli options to run different things

from pipeline.models import models
from pipeline.pipeline import one_dim_x_train
from pipeline.stats import view_cm, get_acc_auc_df, show_outcome_distrib
from pipeline.preprocessing import * # fix later
from pipeline.dataloader import PhonocardiogramAudioDataset, PhonocardiogramByIDDatasetOnlyResult, PhonocardiogramAugmentationTSV
from typing import Tuple
from tqdm import tqdm

def pipeline(X,y):
    # To verify if there's potential need for balancing the dataset
    # show_outcome_distrib(y) 
    acc_list, auc_list, cm_list = one_dim_x_train(X, y, models=models,test_size=0.2, random_state=0)
    view_cm(models, cm_list)
    
    my_df = get_acc_auc_df(models, acc_list, auc_list)
    print(my_df)
    
    
if __name__ == "__main__":
    from pathlib import Path
    from torch.utils.data import DataLoader
    import torch
    import re, os, random
    
    file = Path(".") / "assets" / "the-circor-digiscope-phonocardiogram-dataset-1.0.3"
    # Training On CSV data
    # original_data = pd.read_csv(str(file  / "training_data.csv"))
    
    # model_df = data_wrangling(original_data)
    # X = one_hot_encoding(model_df, ['Sex', 'Murmur', 'Age', 'Systolic murmur quality', 'Systolic murmur pitch',
    #                     'Systolic murmur grading', 'Systolic murmur shape', 'Systolic murmur timing', 'Most audible location'
    #                     ])
    
    # y = model_df['Outcome']



    # Training on actual patient audio files

    def compose_feature_label(
        file : str, 
        lookup_table : PhonocardiogramByIDDatasetOnlyResult, 
        feature_fns : callable,
        transform : callable,
    ) -> Tuple[np.ndarray, int]:
        
        # assume feature_fn will return 1xN array
        audio_ary, _ = librosa.load(file)
        audio_ary = transform(audio_ary) # augmentation on random
        features = np.array([])

        for feature_fn in feature_fns:
            features = np.concatenate( (features, feature_fn(audio_ary)), axis=0)

        return features, int(lookup_table[file])
    
    def beat_based_augmentation(
            data: np.ndarray, 
            file : str,
            seg_tale : PhonocardiogramAugmentationTSV,
            window_length : float = 5.
        ):
        
        match = re.match(r'(\d+)', os.path.basename(file))
        key = int(match.group(1))# for runtime, I won't do error check
        
        seg_content = seg_tale[file]
        
        def find_start_end_tuple():
            first = seg_content[0]
            last = seg_content[-1]
            
            start = first[1][int(first[0] == 0)] # id 0 meaning noise -> 1
            end = last[1][int(last[0] != 0)] # id 0 meaning noise -> 0 (everything after is noise)
            return start, end
        
        start, end = find_start_end_tuple()
        
        window_start = random.uniform(start, end - window_length)
        window_end = window_start + window_length
        
        window_start, window_end = int(window_start * 4000), int(window_end * 4000)
        
        return data[window_start : window_end]
        
        
    segmentation_table = PhonocardiogramAugmentationTSV(file / "training_data")
        
    lookup = PhonocardiogramByIDDatasetOnlyResult(str(file / "training_data.csv"))
    dset = PhonocardiogramAudioDataset(
        file / "training_data",
        ".wav",
        "*", # Everything
        transform=lambda f : compose_feature_label(
            f,
            lookup, 
            [feature_mfcc, feature_chromagram, feature_melspectrogram],
            lambda ary_data : beat_based_augmentation(ary_data, f, segmentation_table)
        )
    )

    loader = DataLoader(
        dset, 
        batch_size=1,
        # collate_fn=lambda x : x,
    )
    X = []
    y = []
    for i in tqdm(loader): # very slow 
        X_i,y_i = i
        X.append(X_i)
        y.append(y_i)
        
    
    # Creating 1 large matrix to train with classical models
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)

    # Training Pipeline
    pipeline(X,y)
    
    