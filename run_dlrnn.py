# from pipeline.utils import energy_band_augmentation_random_win
from pipeline.dl_models import LSTM
from pipeline.preprocessing import feature_mfcc, feature_bandpower_struct, remove_high_frequencies
from pipeline.dataloader import PhonocardiogramAudioDataset, PhonocardiogramByIDDatasetOnlyResult
from pipeline.utils import compose_feature_label, audio_random_windowing

from tqdm import tqdm
import numpy as np
import librosa

if __name__ == "__main__":
    from pathlib import Path
    from torch.utils.data import DataLoader
    import torch
    import torch.nn as nn
    import torch.optim as optim

    file = Path(".") / "assets" / "the-circor-digiscope-phonocardiogram-dataset-1.0.3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training on actual patient audio files
    
    def augmentation(data :np.ndarray, sr : int=4000, window_length_hz :int =200, window_len_sec :float=5.) ->np.ndarray:
        x = data
        x = audio_random_windowing(x, window_len_sec)
        return x

    lookup = PhonocardiogramByIDDatasetOnlyResult(str(file / "training_data.csv"))

    # Feature functions
    features_fn = [
        feature_mfcc, 
        # feature_chromagram, 
        # feature_melspectrogram,
        feature_bandpower_struct(4000,200,0.7),
        # NMF, # found -> takes around 0.1s per file
    ]    
    
    def dset_trans(f : str): # each takes ~0.1s
        # result = compose_feature_label(
        #     f,
        #     lookup, 
        #     features_fn,
        #     lambda ary_data : remove_high_frequencies(augmentation(ary_data,4000,200,3.), sample_rate=4000,cutoff_frequency=450).real,
        #     dim=2,
        #     is_np=False
        # )
        result, _ = librosa.load(f)
        result = augmentation(result, sr= 4000, window_length_hz=200, window_len_sec= 3.)
        result = remove_high_frequencies(result, sample_rate=4000,cutoff_frequency=450).real
        return result, int(lookup[f])
    



    model = LSTM(
        input_dim=3*4000,
        hidden_dim=64,
        output_dim=1,
        num_layers=2
    )

    
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
    
    train_loader = DataLoader(train_dataset, batch_size=64,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64,shuffle=False)

    # training
    model = model.to(device).float()
    
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    num_epoch = 20

    model.train()
    for epoch in range(num_epoch):
        for X,y in tqdm(train_loader):
            X = X.to(device).float()
            y = y.to(device).float()

            optimizer.zero_grad()
            
            out = model(X) # data not stateful
            loss = criterion(out.squeeze(), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}')
        
    # Testing
    model.eval()
    
    with torch.no_grad():
        acc = []
        for Xtest, ytest in tqdm(train_loader):
            Xtest = Xtest.to(device).float()
            ytest = ytest.to(device).float()


            out, _ = model(Xtest)
            pred = (out.squeeze() > 0.5).float()  # Convert probabilities to binary predictions
            
            accu = (pred == ytest).float().mean().item()
            acc.append(accu)
        print(f'Training set Accuracy: {sum(acc)/len(acc):.4f}')
        acc = []
        for Xtest, ytest in tqdm(test_loader):
            Xtest = Xtest.to(device).float()
            ytest = ytest.to(device).float()


            out, _ = model(Xtest)
            pred = (out.squeeze() > 0.5).float()  # Convert probabilities to binary predictions
            accu = (pred == ytest).float().mean().item()
            acc.append(accu)
        print(f'Testing Accuracy: {sum(acc)/len(acc):.4f}')
            