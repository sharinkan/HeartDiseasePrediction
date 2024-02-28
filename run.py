# Should use cli options to run different things

from pipeline.models import models
from pipeline.pipeline import one_dim_x_train
from pipeline.stats import view_cm, get_acc_auc_df, show_outcome_distrib
from pipeline.preprocessing import * # fix later
from pipeline.dataloader import PhonocardiogramAudioDataset, PhonocardiogramByIDDatasetOnlyResult, PhonocardiogramAugmentationTSV
from tqdm import tqdm

from pipeline.utils import compose_feature_label, audio_random_windowing, energy_band_augmentation_random_win

def pipeline(X,y):
    # To verify if there's potential need for balancing the dataset
    # show_outcome_distrib(y) 
    acc_list, auc_list, cm_list = one_dim_x_train(X, y, models=models,test_size=0.1, random_state=0)
    # view_cm(models, cm_list)
    
    my_df = get_acc_auc_df(models, acc_list, auc_list)
    print(my_df)
    
    
if __name__ == "__main__":
    from pathlib import Path
    from torch.utils.data import DataLoader
    import torch
    import re

    from sklearn.decomposition import PCA
    
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
        
        # x = audio_random_windowing(x, window_len_sec)
        return x
    

    def feature_csv(file):
        match = re.match(r'(\d+)', os.path.basename(file))
        key = int(match.group(1))

        return X_CSV.loc[original_data["Patient ID"] == key].to_numpy()[0]
    
    def compose_with_csv(file, audio_extracted_features_label):
        feature, y = audio_extracted_features_label
        csv_feat = feature_csv(file)

        return np.concatenate([feature, csv_feat], axis=0), y

    import random

    features_fn = [
        feature_mfcc,
        # feature_chromagram, 
        # feature_melspectrogram,
        # feature_bandpower_struct(4000,200,0.7),
    ]
    # random.seed(None)
    # features_fn = random.choices(features_fn, k = 2,)

    print([f.__qualname__ for f in features_fn])
        
    lookup = PhonocardiogramByIDDatasetOnlyResult(str(file / "training_data.csv"))


    dset = PhonocardiogramAudioDataset(
        file / "clear_training_data",
        ".wav",
        "*", # Everything
        transform=lambda f : compose_feature_label(
        # transform=lambda f : compose_with_csv(f, compose_feature_label(
            f,
            lookup, 
            features_fn,
            lambda ary_data : remove_high_frequencies(augmentation(ary_data,4000,200,3.), sample_rate=4000,cutoff_frequency=450).real
            # )
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

    n_components = 5
    
    
    for resample in range(BATCHING := 1):
        for i in tqdm(loader): # very slow 
            X_i,y_i = i
            X.append(X_i)
            y.append(y_i)


        
    # Creating 1 large matrix to train with classical models
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)

    # CSV_PORTION = X[:, -30:]
    # FEAT = X[:, :-30]
    
    # pca = PCA(n_components=n_components)
    # data_pca = pca.fit_transform(FEAT)

    # combined_array = np.concatenate((data_pca, CSV_PORTION),axis=1)

    # # Training Pipeline
    # pipeline(combined_array,y)

        
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(X)

    combined_array = np.concatenate((data_pca, CSV_PORTION),axis=1)

    # Training Pipeline
    pipeline(combined_array,y)

    
    