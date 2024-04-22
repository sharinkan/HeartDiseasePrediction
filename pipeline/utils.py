"""Util functions for data augmentation and helpers
"""

import numpy as np
import random
import librosa
from typing import Iterable, Tuple, Literal, Generator, Dict, Union
try:
    from .dataloader import PhonocardiogramByIDDatasetOnlyResult, PhonocardiogramAugmentationTSV
except ImportError:
    from dataloader import PhonocardiogramByIDDatasetOnlyResult, PhonocardiogramAugmentationTSV
    

def energy_band_augmentation(
    audio_data: np.ndarray, 
    sr : int, 
    window_hz_length: int, 
    window_hz_step: int, 
    window_hz_range: tuple
) -> Generator[np.ndarray, None, None]:
    """create augmentation based on frequencies of the audios\
        Example : (100hz - 300hz), (200hz - 400hz)...

    Args:
        audio_data (np.ndarray): audio array
        sr (int): sample rate
        window_hz_length (int): window length by hz
        window_hz_step (int): window step by hz
        window_hz_range (tuple): window range by hz

    Yields:
        Generator[np.ndarray, None, None]: pieces of the audio split on frequency bands
    """
    # fft to frequency domain
    fourier_transform = np.fft.fft(audio_data) 
    frequencies = np.fft.fftfreq(len(fourier_transform), 1 / sr) 
    
    # Steps to extract the freq range of the audios
    abs_freq = np.abs(frequencies)
    win_start, win_end = window_hz_range
    
    for start in range(win_start, win_end - window_hz_step + 1, window_hz_step):
        copy = fourier_transform.copy()
        copy[ ~((abs_freq > start) & (abs_freq < (start + window_hz_length))) ] = 0
        yield copy
    return

    
def sliding_window_iter(start: int, end: int, length: int, overlap: float = 0.4) -> Generator[Tuple[int, int], None, None]:
    """helper function to step through a range

    Args:
        start (int): starting of range
        end (int): end of range
        length (int): each window piece length
        overlap (float, optional): how much should window overlap each other. Defaults to 0.4.

    Yields:
        Generator[Tuple[int, int], None, None]: each window pieces
    """

    step = int((1 - overlap) * length)
    for window_start in range(start, end - length + 1, step):
        yield window_start, window_start + length
    return


def audio_random_windowing(audio : np.ndarray, window_length_sec : float = 5. , sr = 4000) -> np.ndarray:
    """random windowing for audio data

    Args:
        audio (np.ndarray): audio array
        window_length_sec (float, optional): window length in seconds. Defaults to 5..
        sr (int, optional): sample rate. Defaults to 4000.

    Returns:
        np.ndarray: random portion of the audio array 
    """

    
    window_start, window_end = random_windowing(0, len(audio), window_length_sec * sr)
    window_start, window_end = int(window_start), int(window_end)
    
    return audio[window_start : window_end]



def random_windowing(start: int, end : int, length : int) -> Tuple[int, int]:
    """give a random range within start to end

    Args:
        start (int): start of range
        end (int): end of range
        length (int): length of window

    Returns:
        Tuple[int, int]: window
    """

    length = min(end - start , length)
    
    window_start = random.uniform(start, end - length)
    window_end = window_start + length
    
    return window_start, window_end

def energy_band_augmentation_random_win(
    audio_data: np.ndarray, 
    sr : int, 
    window_hz_length: int,
) -> np.ndarray:
    """energy band augmentation with random windowing

    Args:
        audio_data (np.ndarray): audio array
        sr (int): sample rate
        window_hz_length (int): window length in hz

    Returns:
        np.ndarray: augmentation result by random frequency band
    """
    fourier_transform = np.fft.fft(audio_data)
    frequencies = np.fft.fftfreq(len(fourier_transform), 1 / sr) 
    
    # Steps to extract the freq range of the audios
    abs_freq = np.abs(frequencies)
    win_start, win_end = random_windowing(abs_freq[0], sr//2 , window_hz_length)
    
    copy = fourier_transform.copy()
    copy[ ~((abs_freq > win_start) & (abs_freq < win_end)) ] = 0 # remove all frequency data outside the window
    
    return copy



# caching..

from functools import cache

librosa_load_cached = cache(librosa.load)
from time import time
def compose_feature_label(
    file : str, 
    lookup_table : PhonocardiogramByIDDatasetOnlyResult, 
    feature_fns : callable,
    transform : callable,
    dim : Literal[1,2] = 1,
    is_np : bool = True, # only optional at 2D array -> force to numpy array
) -> Tuple[np.ndarray, int]:
    """concatenate features in one array

    Args:
        file (str): audio file
        lookup_table (PhonocardiogramByIDDatasetOnlyResult): results table class instance
        feature_fns (callable): functions to extract feature from audio
        transform (callable): transformation to apply on audio array before applying feature function

    Returns:
        Tuple[np.ndarray, int]: combined features
    """
    
    # assume feature_fn will return 1xN array
    audio_ary, _ = librosa_load_cached(file)
    
    audio_ary = transform(audio_ary)
    features = np.array([])

    if dim == 1:
        for feature_fn in feature_fns:
            features = np.concatenate( (features, feature_fn(audio_ary)), axis=0)
    if dim == 2:
        
        features = [feature_fn(audio_ary) for feature_fn in feature_fns]
        # features = []
        # for feature_fn in feature_fns:
        #     s = time()
        #     features.append(feature_fn(audio_ary))
        #     print("feat ", feature_fn.__qualname__, time() - s)

        features = np.array(features, dtype=object) if is_np else features

    return features, int(lookup_table[file])


def ensemble_methods(models : "MLmodels", X :np.ndarray, option : Literal["hard", "soft"] = "hard") -> np.ndarray:
    """ensemble methods on multiple models

    Args:
        models (MLmodels): list of ML models
        X (np.ndarray): training matrix
        option (Literal["hard", "soft"], optional): voting method option. Defaults to "hard".

    Returns:
        np.ndarray: result array
    """

    arg_max_models = [model for model in models if hasattr(model, "decision_function")] # [N]
    prob_models = [model for model in models if not(hasattr(model, "decision_function"))] # [N, output]

    def hard_voting() -> np.ndarray:
        Ys = []
        for model in models:
            Ys.append(model.predict(X))

        Ys = np.stack(Ys)

        hard_voting_result = np.sum(Ys, axis=0) >= (Ys.shape[0] / 2)
        return hard_voting_result.astype(int)
    
    def soft_voting() -> np.ndarray:
        probs = []
        for model in arg_max_models:
            probs.append(model.decision_function(X))

        for model in prob_models:
            probs.append(model.predict_proba(X)[:, 1]) # second col

        probs = np.stack(probs)

        soft_voting_result = np.average(probs, axis=0) >= 0.5
        return soft_voting_result.astype(int)
    
    return {
        "hard" : hard_voting,
        "soft" : soft_voting
    }[option]()


def ensemble_methods_mixture(models_feat : Dict["MLmodel" , "feature_name"], X_feat : Dict["feature_name", np.ndarray], option : Literal["hard", "soft"] = "hard") -> np.ndarray:
    """ensemble methods with model trained on different feature sets

    Args:
        models_feat (Dict["MLmodel" , "feature_name"]): {Model : feature_key}
        X_feat (Dict["feature_name", np.ndarray]): {feature_key : training data}
        option (Literal["hard", "soft"], optional): voting option. Defaults to "hard".

    Returns:
        np.ndarray: result array
    """
    

    arg_max_models = {}
    prob_models = {} # [N, output]

    for model, feat_name in models_feat.items():
        if hasattr(model, "decision_function"):
            arg_max_models[model] = feat_name
        else:
            prob_models[model] = feat_name

    def hard_voting() ->np.ndarray:
        Ys = []
        for model, feat_name in models_feat.items():
            X = X_feat[feat_name]
            Ys.append(model.predict(X))

        Ys = np.stack(Ys)

        hard_voting_result = np.sum(Ys, axis=0) >= (Ys.shape[0] / 2)
        return hard_voting_result.astype(int)
    
    def soft_voting() ->np.ndarray:
        probs = []
        for model, feat_name in arg_max_models.items():
            X = X_feat[feat_name]
            probs.append(model.decision_function(X))

        for model, feat_name in prob_models.items():
            X = X_feat[feat_name]
            probs.append(model.predict_proba(X)[:, 1]) # second col

        probs = np.stack(probs)

        soft_voting_result = np.average(probs, axis=0) >= 0.5
        return soft_voting_result.astype(int)
    
    return {
        "hard" : hard_voting,
        "soft" : soft_voting
    }[option]()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import librosa
    
    def display_fft(audio_data, sampling_rate):
        # Perform Fourier Transform
        fourier_transform = np.fft.fft(audio_data)
        frequencies = np.fft.fftfreq(len(fourier_transform), 1 / sampling_rate)

        plt.figure(figsize=(10, 4))
        plt.plot(frequencies, np.abs(fourier_transform))
        plt.title("Frequency Domain Representation")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.show()


    audio_data, sampling_rate = librosa.load(
        "assets/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/85334_MV.wav",
        sr=None,
    )

    modified_freq = energy_band_augmentation_random_win(audio_data, sampling_rate, 4000)
    modified_audio = np.fft.ifft(modified_freq).real
    display_fft(modified_audio, sampling_rate)