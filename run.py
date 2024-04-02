# Should use cli options to run different things

from pipeline.models import models
from pipeline.pipeline import one_dim_x_train, mixture_one_dim_x_train
from pipeline.stats import view_cm, get_acc_auc_df, show_outcome_distrib
from pipeline.preprocessing import * # fix later
from pipeline.dataloader import PhonocardiogramAudioDataset, PhonocardiogramByIDDatasetOnlyResult, PhonocardiogramAugmentationTSV
from tqdm import tqdm

from pipeline.utils import compose_feature_label, audio_random_windowing, energy_band_augmentation_random_win, ensemble_methods, ensemble_methods_mixture


VOTING = "hard" 
def pipeline(X,y):
    # To verify if there's potential need for balancing the dataset
    # show_outcome_distrib(y) 
    acc_list, auc_list, cm_list, (test_X, test_Y) = one_dim_x_train(X, y, models=models,test_size=0.1, random_state=0)
    # view_cm(models, cm_list)
    
    my_df = get_acc_auc_df(models, acc_list, auc_list)
    print(my_df)

    # ens
    ens_Y = ensemble_methods(models, test_X, option=VOTING)
    print("ensemble_methods :" , f"{VOTING=}" , np.sum(ens_Y == np.array(test_Y)) / len(test_Y))

def pipeline_mixture(Xs,ys, models):
    acc_list, auc_list, cm_list, (test_X, test_Y), models = mixture_one_dim_x_train(Xs, ys, models_feat=models,test_size=0.1,grid_search_enabled = True)

    # view_cm(models, cm_list)
    
    my_df = get_acc_auc_df(models, acc_list, auc_list)
    print(my_df)

    # ens
    ens_Y = ensemble_methods_mixture(models, test_X, option=VOTING)
    test_Y = test_Y[list(test_Y.keys())[0]]

    print("ensemble_methods :" , f"{VOTING=}" ,np.sum(ens_Y == np.array(test_Y)) / len(test_Y))
    
if __name__ == "__main__":
    from pathlib import Path
    from torch.utils.data import DataLoader
    import torch
    import re, random

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
        match = re.match(r'(\d+)', os.path.basename(file))
        key = int(match.group(1))

        return X_CSV.loc[original_data["Patient ID"] == key].to_numpy()[0]
    
    def compose_with_csv(file, audio_extracted_features_label):
        feature, y = audio_extracted_features_label
        csv_feat = feature_csv(file)

        return np.concatenate([feature, csv_feat], axis=0), y

        
    lookup = PhonocardiogramByIDDatasetOnlyResult(str(file / "training_data.csv"))


    if True: # fix here cli -> using mixture feature
        # Random to be fixed is required at this section
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB

        features_Xfn = {
            "f1" : [
                feature_mfcc,
                # feature_chromagram, 
                # feature_melspectrogram,
                # feature_bandpower_struct(4000,200,0.7),
                # NMF,
            ],
            "f2" : [
                # feature_mfcc,
                # feature_chromagram, 
                # feature_melspectrogram,
                feature_bandpower_struct(4000,200,0.7), ############## 
                # NMF,
            ], # do here
            "f3" : [
                # feature_mfcc,
                # feature_chromagram, 
                # feature_melspectrogram,
                # feature_bandpower_struct(4000,200,0.7),
                NMF,
                    ],
        }
        feature_X = {}
        feature_y = {}
        feature_models = {
            LogisticRegression(solver='liblinear') : "f1",
            SVC() : "f1",
            # SVC() : "f1",
            KNeighborsClassifier(n_neighbors=7) : "f1",
            DecisionTreeClassifier() : "f1",
            RandomForestClassifier() : "f1",
            GaussianNB() : "f1",
        }

        for model,f_name in feature_models.items():
            model.feature_train_name = f"{model}_{f_name}"

        for feat_name, feat_fns in features_Xfn.items():
            ds = PhonocardiogramAudioDataset(
                file / "clear_training_data",
                ".wav",
                "*", # Everything
                transform=lambda f : compose_feature_label(
                    f,
                    lookup, 
                    feat_fns,
                    lambda ary_data : remove_high_frequencies(augmentation(ary_data,4000,200,3.), sample_rate=4000,cutoff_frequency=450).real
                ),
                balancing=True,
                csvfile=str(file / "training_data.csv"),
                shuffle=False
            )
            load = DataLoader(
                ds,
                batch_size=1,
                shuffle=False
                # collate_fn=lambda x : x,
            )
            X = []
            y = []

            for resample in range(BATCHING := 1):
                for i in tqdm(load): # very slow 
                    X_i,y_i = i
                    X.append(X_i)
                    y.append(y_i)
            X = torch.cat(X, dim=0)
            y = torch.cat(y, dim=0)
            feature_X[feat_name] = X
            feature_y[feat_name] = y

        pipeline_mixture(feature_X,feature_y, feature_models)
        exit()

    features_fn = [
        # feature_mfcc, 
        # feature_chromagram, 
        # feature_melspectrogram,
        # feature_bandpower_struct(4000,200,0.7),
        # NMF,
    ]

    print([f.__qualname__ for f in features_fn])
    dset = PhonocardiogramAudioDataset(
        file / "clear_training_data",
        ".wav",
        "*", # Everything
        transform=lambda f : compose_feature_label(
            f,
            lookup, 
            features_fn,
            lambda ary_data : remove_high_frequencies(augmentation(ary_data,4000,200,3.), sample_rate=4000,cutoff_frequency=450).real
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
    pipeline(X,y)
    pipeline(X,y)
