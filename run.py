# Should use cli options to run different things

from pipeline.models import models, param_grids, get_cnn_model
from pipeline.pipeline import one_dim_x_train,cnn_train
from pipeline.stats import view_cm, get_acc_auc_df, show_outcome_distrib
from pipeline.preprocessing import * # fix later
from pipeline.dataloader import PhonocardiogramAudioDataset, PhonocardiogramByIDDatasetOnlyResult, PhonocardiogramAugmentationTSV

from tqdm import tqdm
from pipeline.utils import compose_feature_label, audio_random_windowing, energy_band_augmentation_random_win
import sys
import itertools
import csv
from datetime import datetime

def pipeline(X,y):
    # To verify if there's potential need for balancing the dataset
    # show_outcome_distrib(y) 
    acc_list, auc_list, cm_list = one_dim_x_train(X, y, models=models, param_grids=param_grids, test_size=0.2, random_state=0)
    view_cm(models, cm_list)
    
    # my_df = get_acc_auc_df(models, acc_list, auc_list)
    # print(my_df)

  
if __name__ == "__main__":
    from pathlib import Path
    from torch.utils.data import DataLoader
    import torch
    import re   
    
    file = Path(".") / "assets" / "the-circor-digiscope-phonocardiogram-dataset-1.0.3"
    # Training On CSV data
    original_data = pd.read_csv(str(file  / "training_data.csv"))
    
    model_df = data_wrangling(original_data)
    X_CSV = one_hot_encoding(model_df, [
        'Murmur', 
        'Systolic murmur quality', 
        'Systolic murmur pitch',
        'Systolic murmur grading', 
        'Systolic murmur shape', 
        'Systolic murmur timing',
        'Diastolic murmur quality', 
        'Diastolic murmur pitch',
        'Diastolic murmur grading', 
        'Diastolic murmur shape', 
        'Diastolic murmur timing',
    ])
    y_CSV = model_df['Outcome']



    # Training on actual patient audio files
    segmentation_table = PhonocardiogramAugmentationTSV(file / "training_data")

    def augmentation(data, sr=4000, window_length_hz=200, window_len_sec =5.):
        # This augmentation WILL conflict with new feature of frequency based extraction. ->
        x = data
        # x = energy_band_augmentation_random_win(x, sr=sr, window_hz_length=window_length_hz)
        # x = np.fft.ifft(x).real
        
        x = audio_random_windowing(x, window_len_sec)
        return x



    def feature_csv(file):
        match = re.match(r'(\d+)_(AV|TV|MV|PV|Phc)', os.path.basename(file))
        key = int(match.group(1))
        record = X_CSV.loc[original_data["Patient ID"] == key].to_numpy()[0]
        # channel = match.group(2)
        # channels = ['AV', 'TV', 'MV', 'PV', 'Phc']
        # encoded_channel = np.zeros(len(channels), dtype=int)    
        # if channel in channels:
        #     encoded_channel[channels.index(channel)] = 1
        # record =  np.append(record,encoded_channel)
        return record   
        # return X_CSV.loc[original_data["Patient ID"] == key].to_numpy()[0]   

    def compose_with_csv(file, audio_extracted_features_label):
        feature, y = audio_extracted_features_label
        csv_feat = feature_csv(file)
        return np.concatenate([feature, csv_feat], axis=0), y

    import random

    features_fn = [
        feature_mfcc,
        feature_chromagram, 
        feature_melspectrogram,
        feature_bandpower_struct(4000,200,0.7),
    ]
    # random.seed(None)
    # features_fn = [feature_melspectrogram]

    date_time = datetime.now().strftime("%m%d_%H%M")
    run_name = "cnn_4_features"
    file_path = f'output/{date_time}_{run_name}_output.csv'

    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['accuracy_score', 'auc', 'f1', 'feature_combo'])
    # for r in range(1, len(features_fn)+1):
    for r in range(1, 2):
        for feature_combo in itertools.combinations(features_fn, r):
    
            lookup = PhonocardiogramByIDDatasetOnlyResult(str(file / "training_data.csv"))
            dset = PhonocardiogramAudioDataset(
                file / "training_data",
                ".wav",
                "*", # Everything
                transform=lambda f : compose_with_csv(f, compose_feature_label(
                    f,
                    lookup, 
                    feature_combo,
                    lambda ary_data : augmentation(ary_data,4000,300,3.))
                ),  
                balancing=True,
                csvfile=str(file / "training_data.csv"),
                shuffle=True
            )

            loader = DataLoader(
                dset, 
                batch_size=1,
                shuffle=True
                # collate_fn=lambda x : x,
            )
            X = []
            y = []

            for resample in range(BATCHING := 1):
                for i in tqdm(loader): # very slow 
                    X_i,y_i = i
                    X.append(X_i)
                    y.append(y_i)

            # Creating 1 large matrix to train with classical models
            X = torch.cat(X, dim=0)
            y = torch.cat(y, dim=0)

            # Training Pipeline
            # pipeline(X,y)
            print('~'*10)
            print(X.shape[1])
            acc, auc, f1 = cnn_train(X,y)
            r = [acc, auc, f1] + list(feature_combo)
            with open(file_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(r)

    
    
