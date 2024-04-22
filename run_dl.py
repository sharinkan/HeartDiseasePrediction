from pipeline.dl_models import *
from pipeline.preprocessing import * # fix later
from pipeline.dataloader import PhonocardiogramAudioDataset, PhonocardiogramByIDDatasetOnlyResult, PhonocardiogramAugmentationTSV
from pipeline.utils import compose_feature_label, audio_random_windowing, energy_band_augmentation_random_win, ensemble_methods, ensemble_methods_mixture

from tqdm import tqdm
import numpy as np
from typing import Dict


if __name__ == "__main__":
    from pathlib import Path
    from torch.utils.data import DataLoader
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import re, random

    file = Path(".") / "assets" / "the-circor-digiscope-phonocardiogram-dataset-1.0.3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training on actual patient audio files
    
    def augmentation(data :np.ndarray, sr : int=4000, window_length_hz :int =200, window_len_sec :float=5.) ->np.ndarray:
        x = data
        # x = energy_band_augmentation_random_win(x, sr=sr, window_hz_length=window_length_hz)
        # x = np.fft.ifft(x).real
        x = audio_random_windowing(x, window_len_sec)
        return x

    lookup = PhonocardiogramByIDDatasetOnlyResult(str(file / "training_data.csv"))

    # Feature functions
    features_fn = [
        feature_mfcc, 
        # feature_chromagram, 
        # feature_melspectrogram,
        feature_bandpower_struct(4000,200,0.7),
        NMF,
    ]    
    
    def dset_trans(f : str):
        return compose_feature_label(
            f,
            lookup, 
            features_fn,
            lambda ary_data : remove_high_frequencies(augmentation(ary_data,4000,200,3.), sample_rate=4000,cutoff_frequency=450).real,
            dim=2,
            is_np=False
        )
        
    def create_MLPs():
        rand_sample = np.random.random((4000 * 10,)) # 10 sec sample for 4000sr
        feature_space = [f_fn(rand_sample) for f_fn in features_fn]
        
        feat_sizes = [feat_matx.shape[0] for feat_matx in feature_space]
        mlps = [
            MLP([
                feat_size, 
                64,
                64 * 2, 
                1,] , torch.nn.ReLU)
            for feat_size in feat_sizes
        ]
        
        return mlps, feature_space
        
        
    feature_based_mlps, example_feature = create_MLPs()
    combinedMLP = CombinedMLP(feature_based_mlps)
    
    print([f.__qualname__ for f in features_fn])
    dset = PhonocardiogramAudioDataset(
        file / "clear_training_data",
        ".wav",
        "*", # Everything
        transform=dset_trans,
        balancing=True,
        csvfile=str(file / "training_data.csv"),
        shuffle=True,
    )
    
    train_size = int(0.8 * len(dset))
    test_size = len(dset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128,shuffle=False)

    # training
    combinedMLP.to(device)
    
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(combinedMLP.parameters(), lr=0.00001)

    num_epoch = 5
    
    for epoch in range(num_epoch):
        combinedMLP.train()
        for X,y in tqdm(train_loader):
            X = [x_sub.to(device) for x_sub in X]
            y = y.to(device)

            optimizer.zero_grad()
            out = combinedMLP(X)
            # LATER
            loss = criterion(out.squeeze(), y.float())
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}')
        
    # Testing
    combinedMLP.eval()
    acc = []
    
    with torch.no_grad():
        for Xtest, ytest in tqdm(train_loader):
            Xtest = [x_sub.to(device) for x_sub in Xtest]
            ytest = y.to(device)


            out = combinedMLP(Xtest)
            print(out, ytest)
            pred = (out.squeeze() > 0.5).float()  # Convert probabilities to binary predictions
            accu = (pred == ytest).float().mean().item()
            acc.append(accu)
        print(f'Training set Accuracy: {sum(acc)/len(acc):.4f}')
        
        for Xtest, ytest in tqdm(test_loader):
            Xtest = [x_sub.to(device) for x_sub in Xtest]
            ytest = y.to(device)


            out = combinedMLP(Xtest)
            pred = (out.squeeze() > 0.5).float()  # Convert probabilities to binary predictions
            accu = (pred == ytest).float().mean().item()
            acc.append(accu)
        print(f'Testing Accuracy: {sum(acc)/len(acc):.4f}')
            