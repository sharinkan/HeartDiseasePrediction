from pipeline.dl_models import *
from pipeline.preprocessing import * # fix later
from pipeline.dataloader import PhonocardiogramAudioDataset, PhonocardiogramByIDDatasetOnlyResult
from pipeline.utils import compose_feature_label, audio_random_windowing

from tqdm import tqdm
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from typing import Callable
from datetime import datetime
AUDIO_WINDOW_LEN_SEC = 3.
SAMPLE_RATE = 4000


def log_gradients_in_model(model, logger, step):
    # https://discuss.pytorch.org/t/logging-gradients-on-each-iteration/94256
    for tag, value in model.named_parameters():
        if value.grad is not None:
            logger.add_histogram(tag + "/grad", value.grad.cpu(), step)


def train_nn_model(
        model : nn.Module, 
        dataset_transformation : Callable[[str], list[np.ndarray, int]], 
        batched_X_trans : Callable, # can't type hint properly as it depends on model input
        batched_Y_trans : Callable,
    ) -> None:

    dset = PhonocardiogramAudioDataset(
        file / "clear_training_data",
        ".wav",
        "*", # Everything
        transform=dataset_transformation,
        balancing=True,
        csvfile=str(file / "training_data.csv"),
        shuffle=True,
    )
    
    train_size = int(0.8 * len(dset))
    test_size = len(dset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32,shuffle=False)

    # training
    model = model.to(device)
    
    writer = SummaryWriter(f"ign_runs/summary{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}")

    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3)

    num_epoch = 200


    model.train()
    for epoch in range(num_epoch):
        for indx, (X,y) in enumerate(tqdm(train_loader)):
            X = batched_X_trans(X)
            y = batched_Y_trans(y)

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out.squeeze(dim=1), y.float())

            loss.backward()
            optimizer.step()

            acc = ((out.squeeze(dim=1) > 0.5).float() == y).float().mean().item()

            global_step = epoch * len(train_loader) + indx
            writer.add_scalar("training loss", loss.item(), global_step)
            writer.add_scalar("accuracy", acc, global_step)
            log_gradients_in_model(model, writer, global_step)


        scheduler.step()

    
    # Testing
    print("Start Testing")
    model.eval()
    with torch.no_grad():
        labeled_loaders = {
            "Training" : train_loader, # due augmentation, it's not exactly the training data
            "Testing" : test_loader
        }

        for mode, loader in labeled_loaders.items():
            acc = []
            for Xtest, ytest in tqdm(loader):
                Xtest = batched_X_trans(Xtest)
                ytest = batched_Y_trans(ytest)

                out = model(Xtest)
                pred = (out.squeeze() > 0.5).float()  # Convert probabilities to binary predictions
                accu = (pred == ytest).float().mean().item()
                acc.append(accu)
            print(f'{mode} set Accuracy: {sum(acc)/len(acc):.4f}')


def augmentation(data :np.ndarray, sr : int=SAMPLE_RATE, window_length_hz :int =200, window_len_sec :float=AUDIO_WINDOW_LEN_SEC) ->np.ndarray:
    x = data
    x = audio_random_windowing(x, window_len_sec)
    return x


if __name__ == "__main__":
    file = Path(".") / "assets" / "the-circor-digiscope-phonocardiogram-dataset-1.0.3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lookup = PhonocardiogramByIDDatasetOnlyResult(str(file / "training_data.csv"))


    # MLP
    # Feature functions

    # features_fn = [
    #     feature_mfcc, 
    #     feature_chromagram, 
    #     feature_melspectrogram,
    #     feature_bandpower_struct(SAMPLE_RATE,200,0.7),
    #     # NMF, # found -> takes around 0.1s per file
    # ]
    # print("Features :", [f.__qualname__ for f in features_fn])
    
    # def dataset_trans_concat_feat_fns(f : str): # each takes ~0.1s
    #     result = compose_feature_label(
    #         f,
    #         lookup, 
    #         features_fn,
    #         lambda ary_data : remove_high_frequencies(augmentation(ary_data,SAMPLE_RATE,200,3.), sample_rate=SAMPLE_RATE,cutoff_frequency=450).real,
    #         dim=2,
    #         is_np=False
    #     )
    #     return result
        
    # def create_MLPs():
    #     rand_sample = np.random.random((SAMPLE_RATE * 10,)) # 10 sec sample for 4000sr
    #     feature_space = [f_fn(rand_sample) for f_fn in features_fn]
        
    #     feat_sizes = [feat_matx.shape[0] for feat_matx in feature_space]
    #     mlps = [
    #         MLP([
    #             feat_size,
    #             64,
    #             64 * 2,
    #             64 * 2 * 2,
    #             64 * 2,
    #             64,
    #             1,] , torch.nn.ReLU)
    #         for feat_size in feat_sizes
    #     ]
        
    #     return mlps, feature_space
        
    # feature_based_mlps, _ = create_MLPs()
    # combinedMLP = CombinedMLP(feature_based_mlps)

    # train_nn_model(combinedMLP, dataset_trans_concat_feat_fns, lambda x : [x_sub.to(device) for x_sub in x], lambda y : y.to(device))


    # LSTM 
    lstm_model = LSTM(
        input_dim=int(AUDIO_WINDOW_LEN_SEC) * SAMPLE_RATE,
        hidden_dim=64,
        output_dim=1,
        num_layers=2
    )

    rnn_model = RNN(
        input_dim=int(AUDIO_WINDOW_LEN_SEC) * SAMPLE_RATE,
        hidden_dim=64,
        output_dim=1,
        num_layers=2
    )

    def dataset_trans_raw_audio_remove_high_freq(f : str):
        result, _ = librosa.load(f)
        result = augmentation(result, sr= SAMPLE_RATE, window_length_hz=200, window_len_sec=AUDIO_WINDOW_LEN_SEC)
        result = remove_high_frequencies(result, sample_rate=SAMPLE_RATE,cutoff_frequency=450).real
        return result, int(lookup[f])
    
    train_nn_model(lstm_model, dataset_trans_raw_audio_remove_high_freq, lambda x : x.to(device).float(), lambda y : y.to(device).float())