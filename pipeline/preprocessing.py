"""preprocessing actions such as feature extraction
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import librosa
import numpy as np
from scipy import signal
from .utils import sliding_window_iter
import opensmile
import audiofile
import os
import math
import soundfile
from pathlib import Path
import re
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from pprint import pprint
from sklearn.metrics import roc_auc_score, f1_score

from typing import Tuple

def data_wrangling(df: pd.DataFrame) -> pd.DataFrame:
    """data wrangling on csv labels

    Args:
        df (pd.DataFrame): csv file in dataframe

    Returns:
        pd.DataFrame: result dataframe
    """
    data = df.copy(deep=True)
    data["Age"].fillna(
        "Young Adult", inplace=True
    )  # Labeling 'nan' values in age column

    # New columns
    data["Recording Channels"] = data["Recording locations:"].str.split("+")
    data["Murmur Channels"] = data["Murmur locations"].str.split("+")

    # Mapping of string values to integer values
    outcome_mapping = {"Normal": 0, "Abnormal": 1}
    data["Outcome"] = data["Outcome"].replace(outcome_mapping)

    # Further mapping if required later
    # smt_mapping = {'Early-systolic': 0, 'Mid-systolic': 1, 'Late-systolic': 2, 'Holosystolic': 3}
    # sms_mapping = {'Crescendo': 0, 'Decrescendo': 1, 'Diamond': 2, 'Plateau': 3}
    # smg_mapping = {'I/VI': 0, 'II/VI': 1, 'III/VI': 2}
    # smp_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    # smq_mapping = {'Musical': 0, 'Blowing': 1, 'Harsh': 2}

    # dmt_mapping = {'Early-diastolic': 0, 'Holodiastolic': 1, 'Mid-diastolic': 2}
    # dms_mapping = {'Decrescendo': 0, 'Plateau': 1}
    # dmg_mapping = {'I/IV': 0, 'II/IV': 1, 'III/IV': 2}
    # dmp_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    # dmq_mapping = {'Blowing': 0, 'Harsh': 1}

    return data


def one_hot_encoding(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """one hot encode dataframe based on selected columns

    Args:
        df (pd.DataFrame): original dataframe
        cols (list): selected columns

    Returns:
        pd.DataFrame: encoded data
    """
    data = df.copy(deep=True)
    data = data[cols]
    df_encoded = pd.get_dummies(data, dtype=int)

    return df_encoded


def remove_high_frequencies(audio_data : np.ndarray, sample_rate :int , cutoff_frequency: int) -> np.ndarray:
    """remove audio data above certain frequencies

    Args:
        audio_data (np.ndarray): audio array
        sample_rate (int): sample rate of audio
        cutoff_frequency (int): max allowed frequency

    Returns:
        np.ndarray: result audio
    """
    audio_fft = np.fft.fft(audio_data)
    fft_freqs = np.fft.fftfreq(audio_fft.size, 1 / sample_rate)
    audio_fft[np.abs(fft_freqs) > cutoff_frequency] = 0
    modified_audio = np.fft.ifft(audio_fft)
    return modified_audio


def segment_audio(
    audio_file: str,
    sr: int = 4000,
    start_sample: float = 0,  # In Sec
    end_sample: float = None,  # In Sec
    max_freq: int = None,
    n_mels: int = None,
    win_length: int = 128,  # (0,127)
    hop_length: int = 20,  # (0,127) , (20,147), ...
    to_db: bool = False,  # to decible
) -> np.ndarray:
    """
    Segments and extracts features from an audio file.

    Args:
        audio_file (str): Path to the audio file.
        sr (int): Sampling rate.
        start_sample (float): Start sample for segmentation.
        end_sample (float): End sample for segmentation. If None, the entire audio is used.
        max_freq (int): Maximum frequency to retain in the mel spectrogram. If None, no cutoff.
        n_mels (int): Number of mel frequency bands. If None, use librosa's default.
        win_length (int): Size of the FFT window in samples.
        hop_length (int): Number of samples between successive frames.
        to_db (bool): Whether to convert the spectrogram to decibels.

    Returns:
        np.ndarray: Mel spectrogram or decibel-scaled mel spectrogram.
    """
    sample, _ = librosa.load(
        audio_file, sr=sr, offset=start_sample, duration=end_sample
    )

    if max_freq:
        sample = remove_high_frequencies(sample, sr, max_freq).real

    mel_spec = librosa.feature.melspectrogram(
        y=sample, sr=sr, n_mels=n_mels, win_length=win_length, hop_length=hop_length
    )

    if to_db:
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec


def feature_mfcc(waveform :np.ndarray, sample_rate=4000, n_mfcc=42) -> np.ndarray:
    """Compute the MFCCs for all STFT frames and get the mean of each column of the resulting matrix to create a feature array\
        40 filterbanks = 40 coefficients\
            https://github.com/IliaZenkov/sklearn-audio-classification/blob/master/sklearn_audio_classification.ipynb
    Args:
        waveform (np.ndarray): audio
        sample_rate (int, optional): audio sample rate. Defaults to 4000.
        n_mfcc (int, optional): number of mfcc features. Defaults to 42.

    Returns:
        np.ndarray: feature result from mfcc
    """
    mfc_coefficients = np.mean(
        librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0
    )
    return mfc_coefficients


def feature_chromagram(waveform : np.ndarray, sample_rate=4000) -> np.ndarray:
    """STFT computed here explicitly; mel spectrogram and MFCC functions do this under the hood

    Args:
        waveform (np.ndarray): audio
        sample_rate (int, optional): audio sample rate. Defaults to 4000.

    Returns:
        np.ndarray: chromagram feature
    """
    stft_spectrogram = np.abs(librosa.stft(waveform))
    # Produce the chromagram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    chromagram = np.mean(
        librosa.feature.chroma_stft(S=stft_spectrogram, sr=sample_rate).T, axis=0
    )
    return chromagram


def feature_melspectrogram(waveform : np.ndarray, sample_rate=4000, n_mels=16) -> np.ndarray:
    """_summaProduce the mel spectrogram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array\
        Using 8khz as upper frequency bound should be enough for most speech classification tasksry_

    Args:
        waveform (np.ndarray): audio
        sample_rate (int, optional): audio sample rate. Defaults to 4000.
        n_mels (int, optional): number of melspectrogram features. Defaults to 16.

    Returns:
        np.ndarray: melspectrogram features
    """
    melspectrogram = np.mean(
        librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels).T,
        axis=0,
    )
    return melspectrogram


def feature_mel_2d(waveform: np.ndarray, sample_rate=4000, n_mels=128) -> np.ndarray:
    """Extract 2D mel-spectrogram features from an audio waveform.

    Args:
        waveform (np.ndarray): Audio waveform.
        sample_rate (int, optional): Sample rate of the audio. Defaults to 4000.
        n_mels (int, optional): Number of Mel bands to generate. Defaults to 128.

    Returns:
        np.ndarray: 2D mel-spectrogram features (n_mels, time_frames).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=sample_rate, n_mels=n_mels
    )
    return mel_spec


def bandpower_struct(x : np.ndarray, fs : int) -> callable:
    """bandpower feature function constructor

    Args:
        x (np.ndarray): audio
        fs (int): max audio frequency

    Returns:
        callable: bandpower feature function
    """
    f, Pxx = signal.periodogram(x, fs=fs)

    def bandpower(fmin, fmax):
        ind_min = np.argmax(f > fmin) - 1
        ind_max = np.argmax(f > fmax) - 1
        return np.trapz(Pxx[ind_min:ind_max], f[ind_min:ind_max])

    return bandpower


# band power feature
def feature_bandpower_struct(sample_rate : int=4000, interval :int =200, overlap_percentage : float=0.7) -> callable:
    """bandpower feature function constructor for specific audio sample with windowing method on frequency bands

    Args:
        sample_rate (int, optional): audio sample rate (all audio using this should have same sr). Defaults to 4000.
        interval (int, optional): windowing interval on frequency bands . Defaults to 200.
        overlap_percentage (float, optional): overlapping percentage. Defaults to 0.7.

    Returns:
        callable: bandpower feature function
    """
    freq_windows = list(
        sliding_window_iter(0, sample_rate // 2, interval, overlap_percentage)
    )

    def feature_bandpower(waveform :np.ndarray) -> np.ndarray:
        """give bandpower feature on audio

        Args:
            waveform (np.ndarray): audio

        Returns:
            np.ndarray: bandpower feature
        """
        bandpower_func = bandpower_struct(waveform, fs=sample_rate // 2)
        features = []
        for win_start, win_end in freq_windows:
            features.append(bandpower_func(win_start, win_end))

        return features

    return feature_bandpower


class features_csv:
    
    initialized: bool = False
    X_CSV: pd.DataFrame = None
    y_CSV: pd.Series = None
    IDs: pd.Series = None

    def init(dataset_root):
        if features_csv.initialized:
            return
        
        features_csv.initialized = True
        
        # feature from the csv file, transformed by one-hot encoder
        file = Path(dataset_root)
        # Training On CSV data
        original_data = pd.read_csv(str(file / "training_data.csv"))
        
        model_df = data_wrangling(original_data)
        features_csv.X_CSV = one_hot_encoding(model_df, [
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
        features_csv.y_CSV = model_df['Outcome'].apply(int).to_list()
        features_csv.IDs = model_df['Patient ID'].apply(int).to_list()

    def get_by_id(patient_id):
        features_csv.IDs: list[int]
        features_csv.X_CSV: pd.DataFrame
        assert features_csv.initialized, "features_csv is not initialized, need to call features_csv.init first!"
        idx = features_csv.IDs.index(int(patient_id))
        return features_csv.X_CSV.iloc[idx]
    
    def get_for_file(file_path: str):
        match = re.search(r'\d+_', file_path)
        patient_id = match.group()[:-1]
        return features_csv.get_by_id(patient_id)


def build_feature_extractor(use_features, cutoff_frequency = 2000, n_mels = 128, n_mfcc = 84, sample_rate = 4000):

    def extract_features(file):
        # load an individual soundfile
        with soundfile.SoundFile(file) as audio:
            
            # Remove high frequency noise
            if cutoff_frequency > 0:
                waveform = remove_high_frequencies(
                    audio_data = audio.read(dtype="float32"),
                    sample_rate = sample_rate,
                    cutoff_frequency=cutoff_frequency
                ).real
            else:
                waveform = audio.read(dtype="float32")
            
            # compute features of soundfile
            if "mel_2d" in use_features:
                assert len(use_features) == 1, "When using 'mel_2d' feature, it should be the only feature"
                return feature_mel_2d(waveform, sample_rate, n_mels)
            
            else:
                features_from_file = []
            
                if "chromagram" in use_features:
                    features_from_file.append(feature_chromagram(waveform, sample_rate))

                if "melspectrogram" in use_features:
                    features_from_file.append(feature_melspectrogram(waveform, sample_rate, n_mels))
            
                if "mfcc" in use_features:
                    features_from_file.append(feature_mfcc(waveform, sample_rate, n_mfcc))
            
                if "csv" in use_features:
                    features_from_file.append(features_csv.get_for_file(file).to_numpy())

                # stack feature arrays horizontally to create a feature matrix
                return np.hstack(features_from_file)

    return extract_features


class TCDPdata:
    
    def __init__(self, dataset_root):
        self.training_data_path = dataset_root + "/training_data"
        self.filtered_files: list[str] = []
        self.y_dict: dict[str, int] = {}

        
        features_csv.init(dataset_root)

        # create self.filtered_files
        keywords = ['TV','AV','PV','MV']
        extension = '.wav'
        for filename in os.listdir(self.training_data_path):
            if any(keyword in filename for keyword in keywords) and filename.endswith(extension):
                self.filtered_files.append(filename)

        # creat self.y_dict
        dataset_info = pd.read_csv(dataset_root + '/training_data.csv')
        dataset_info['Mapped_Outcome'] = dataset_info['Outcome'].map({
            'Normal': 1,
            'Abnormal': 0
        })
        self.y_dict = dict(zip(dataset_info['Patient ID'], dataset_info['Mapped_Outcome']))

    def getXy(self, extract_features: callable):
        X, y = [], []
        for file_name in tqdm(self.filtered_files):
            file_path = os.path.join(self.training_data_path, file_name)
            feature = extract_features(file_path)
            X.append(feature)

            y.append(self.y_dict[int(
                re.search(r'\d+_', file_name) \
                .group()[:-1]
            )])

        # If the features are 2D mel-spectrograms, pad them to the same shape
        if X and X[0].ndim == 2:
            max_length = max(x.shape[1] for x in X)
            X = [np.pad(x, ((0, 0), (0, max_length - x.shape[1])), mode='constant') for x in X]

        return np.array(X), np.array(y)


def high_dim_min_max_scaler(normalize_axis: tuple[int]):
    
    def min_max_fit_transform(m: np.ndarray):
        n_normalize_axis = len(normalize_axis)

        def swap_follow_normalize_axis(m):
            m_swapped = m.copy()
            for i in range(n_normalize_axis):
                m_swapped = m_swapped.swapaxes(i, normalize_axis[i])
            return m_swapped

        m_swapped = swap_follow_normalize_axis(m)

        operate_axis = tuple(range(n_normalize_axis))
        maxs = m_swapped.max(axis=operate_axis)
        mins = m_swapped.min(axis=operate_axis)
        ranges = maxs - mins
        diffs = m_swapped - mins
        m_swapped_scaled = np.divide(
            diffs,
            ranges,
            out=np.zeros_like(diffs),
            where=(ranges!=0)
        )
        
        m_scaled = swap_follow_normalize_axis(m_swapped_scaled)
        assert round(m_scaled.min()) == 0 and round(m_scaled.max()) == 1, "strict mode test"
        return m_scaled

    return min_max_fit_transform


def high_dim_standard_scaler(normalize_axis: tuple[int]):
    
    def standard_fit_transform(m: np.ndarray):
        n_normalize_axis = len(normalize_axis)

        def swap_follow_normalize_axis(m):
            m_swapped = m.copy()
            for i in range(n_normalize_axis):
                m_swapped = m_swapped.swapaxes(i, normalize_axis[i])
            return m_swapped

        m_swapped = swap_follow_normalize_axis(m)

        operate_axis = tuple(range(n_normalize_axis))
        means = m_swapped.mean(axis=operate_axis)
        stds = m_swapped.std(axis=operate_axis)
        diffs = m_swapped - means
        m_swapped_scaled = np.divide(
            diffs,
            stds,
            out=np.zeros_like(diffs),
            where=(stds!=0)
        )
        
        m_scaled = swap_follow_normalize_axis(m_swapped_scaled)
        assert round(m_scaled.mean(axis=normalize_axis).sum()) == 0, "strict mode test"
        assert round(m_scaled.std(axis=normalize_axis).mean()) == 1, "strict mode test"
        return m_scaled

    return standard_fit_transform


def gen_datesets(features, labels, use_datasets, train_size, random_state, normalize_axis:tuple[int]|int=None) -> tuple[np.ndarray, np.ndarray]:
    """
    normalize_axis: optional. The axis that normalizer will move along.
    e.g. for 2-D feature matrix, rows for samples and columns for features, normalizer moves along axis=0(rows). normalize_axis
    """
    
    features_dim = len(features.shape)
    if features_dim > 2:
        
        if type(normalize_axis) != tuple and type(normalize_axis) != int:
            raise(ValueError, f"Input features matrix is {features_dim}-dimension. Please specify normalize_axis argument.")

        if type(normalize_axis) == int:
            normalize_axis = (normalize_axis, )
            
        max_normalize_axis = max(normalize_axis)
        
        if 0 not in normalize_axis:
            raise ValueError(f"normalize_axis must inclue 0. Data points can not be normalized with different parameters.")
        elif max_normalize_axis > features_dim - 1:
            raise ValueError(f"Input features matrix is {features_dim}-dimension. the max value in normalize_axis should not exceed {features_dim - 1}.")
        else:
            normalizer: dict[str, callable] = {
                "raw": lambda x: x,
                "scaled": high_dim_min_max_scaler(normalize_axis),
                "minmax": high_dim_standard_scaler(normalize_axis),
            }
    else:
        normalizer: dict[str, callable] = {
            "raw": lambda x: x,
            "scaled": StandardScaler().fit_transform,
            "minmax": MinMaxScaler().fit_transform
        }

    y = {
        "train": None,
        "test": None
    }

    X = {
        x_type: y.copy()
        for x_type in use_datasets
    }

    for x_type in use_datasets:
        x = X[x_type]
        x['train'], x['test'], y["train"], y["test"] = train_test_split(
            normalizer[x_type](features), 
            labels, 
            train_size=train_size,
            random_state=random_state
        )

    ## defensive checking
    # assert "scaled" not in use_datasets or -0.3 < X['scaled']['train'][0].mean() < 0.3
    # assert "scaled" not in use_datasets or -0.3 < X['scaled']['test'][0].mean() < 0.3
    # assert "minmax" not in use_datasets or X['minmax']['train'][0].max() == 1
    # assert "minmax" not in use_datasets or X['minmax']['test'][0].min() == 0
    # assert train_size - 0.3 < len(y["train"]) / len(labels) < train_size + 0.3

    return X, y


def cross_train(X, y, model_args, verbose=True):
    
    models = {}
    scores = {}
    
    for x_type in X.keys():
        x = X[x_type]
        
        scores[x_type] = {m_name: {} for m_name in model_args}
        models[x_type] = {m_name: None for m_name in model_args}
        
        for m_name in model_args:
            
            if verbose:
                print(f"Dataset: {x_type}, Model: {m_name}, Training...")
                
            m = model_args[m_name]        
            m_builder: callable = m["class"]
            m_kwargs: dict = m['kwargs']
            
            model = m_builder(**m_kwargs)
            model.fit(x['train'], y['train'])
            models[x_type][m_name] = model
            
            for t in ('train', 'test'):
                y_pred = model.predict(x[t])
                
                scores[x_type][m_name][t] = {
                    "accuracy": model.score(x[t], y[t]),
                    "f1" : f1_score(y[t], y_pred),
                    "auc": roc_auc_score(y[t], y_pred),
                }
                
                if verbose:
                    print(f"Performance on {t} set:")
                    pprint(scores[x_type][m_name][t])

            print()

    return models, scores
        
        
        
def feature_opensmile(
    waveform: np.ndarray,
    sample_rate: int = 4000,
    one_d: bool = False,
    feature_set : str = opensmile.FeatureSet.eGeMAPSv02,
    short: bool = False,
) -> pd.DataFrame:
    """Calculate the opensmile features for each 2 second. Step = 1 sec. there are 25 features in total.
    If the input waveform is x seconds, then the output DataFrame will be x rowx * 25 columns.
    Ref for eGeMAPSv02: https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf

    Args:
        waveform (np.ndarray): audio
        sample_rate (int, optional): audio sample rate. Defaults to 4000.
        one_d (bool, optional): is the data 1 dimensional. Defaults to False.
        feature_set ("FEATURE CODE", optional): feature code. Defaults to opensmile.FeatureSet.eGeMAPSv02.
        short (bool, optional): return subset. Defaults to False.

    Returns:
        pd.DataFrame: features
    """
    smile = opensmile.Smile(
        feature_set=feature_set,
        feature_level=(
            opensmile.FeatureLevel.LowLevelDescriptors
            if one_d is False else
            opensmile.FeatureLevel.Functionals
        )
    )
    features_df = smile.process_signal(waveform, sample_rate)
    if not short:
        return features_df if one_d is False else features_df.to_numpy()[0]
    else:
        return features_df[
            [  # use this
                "F3frequency_sma3nz",
                "F3bandwidth_sma3nz",
                "F2frequency_sma3nz",
                "F2bandwidth_sma3nz",
                "F1frequency_sma3nz",
                "F1bandwidth_sma3nz",
                "F3amplitudeLogRelF0_sma3nz",
                "F1amplitudeLogRelF0_sma3nz",
                "F2amplitudeLogRelF0_sma3nz",
                "logRelF0-H1-A3_sma3nz",
            ]
        ]

def NMF(waveform, S = 3, FRAME = 512, HOP = 256, beta = 2, epsilon = 1e-10, threshold = 0.05, MAXITER = 1000): 

    """
    inputs : 
    --------
        waveform  : The input signal data
        S         : The number of sources to extract
        FRAME     :
        HOP       :
        beta      : Beta divergence considered, default=2 (Euclidean)
        epsilon   : Error to introduce
        threshold : Stop criterion 
        MAXITER   : The number of maximum iterations, default=1000
        display   : Display plots during optimization 
        displayEveryNiter : only display last iteration 
                                                            

    outputs :
    ---------
        
        W : dictionary matrix [KxS], W>=0
        H : activation matrix [SxN], H>=0
        cost_function : the optimised cost function over iterations
        
    Algorithm : 
    -----------

    1) Randomly initialize W and H matrices
    2) Multiplicative update of W and H 
    3) Repeat step (2) until convergence or after MAXITER 

        
    """
    #############
    # Return the complex Short Term Fourier Transform
    sound_stft = librosa.stft(waveform, n_fft = FRAME, hop_length = HOP)

    # Magnitude Spectrogram
    sound_stft_Magnitude = np.abs(sound_stft)

    V = sound_stft_Magnitude + epsilon
    K, N = np.shape(V)

    counter  = 0
    cost_function = []
    beta_divergence = 1

    def divergence(V,W,H, beta = 2):
    
        """
        beta = 2 : Euclidean cost function
        beta = 1 : Kullback-Leibler cost function
        beta = 0 : Itakura-Saito cost function
        """ 
        
        if beta == 0 : return np.sum( V/(W@H) - math.log10(V/(W@H)) -1 )
        
        if beta == 1 : return np.sum( V*math.log10(V/(W@H)) + (W@H - V))
        
        if beta == 2 : return 1/2*np.linalg.norm(W@H-V)


    K, N = np.shape(V)

    # Initialisation of W and H matrices : The initialization is generally random
    W = np.abs(np.random.normal(loc=0, scale = 2.5, size=(K,S)))    
    H = np.abs(np.random.normal(loc=0, scale = 2.5, size=(S,N)))

    # Plotting the first initialization
    while beta_divergence >= threshold and counter <= MAXITER:
        
        # Update of W and H
        H *= (W.T@(((W@H)**(beta-2))*V))/(W.T@((W@H)**(beta-1)) + 10e-10)
        W *= (((W@H)**(beta-2)*V)@H.T)/((W@H)**(beta-1)@H.T + 10e-10)
        
        
        # Compute cost function
        beta_divergence =  divergence(V,W,H, beta = 2)
        cost_function.append( beta_divergence )

        counter +=1

    # if counter -1 == MAXITER : 
    #     print(f"Stop after {MAXITER} iterations.")
    # else : 
    #     print(f"Convergeance after {counter-1} iterations.")
        
    return W.reshape([1, -1])[0]

def NMF_Norm(waveform, S = 3, FRAME = 512, HOP = 256, beta = 2, epsilon = 1e-10, threshold = 0.05, MAXITER = 1000): 

    #############
    # Return the complex Short Term Fourier Transform
    sound_stft = librosa.stft(waveform, n_fft = FRAME, hop_length = HOP)

    # Magnitude Spectrogram
    sound_stft_Magnitude = np.abs(sound_stft)

    V = sound_stft_Magnitude + epsilon
    K, N = np.shape(V)

    counter  = 0
    cost_function = []
    beta_divergence = 1

    def divergence(V,W,H, beta = 2):
    
        """
        beta = 2 : Euclidean cost function
        beta = 1 : Kullback-Leibler cost function
        beta = 0 : Itakura-Saito cost function
        """ 
        
        if beta == 0 : return np.sum( V/(W@H) - math.log10(V/(W@H)) -1 )
        
        if beta == 1 : return np.sum( V*math.log10(V/(W@H)) + (W@H - V))
        
        if beta == 2 : return 1/2*np.linalg.norm(W@H-V)


    K, N = np.shape(V)

    # Initialisation of W and H matrices : The initialization is generally random
    W = np.abs(np.random.normal(loc=0, scale = 2.5, size=(K,S)))    
    H = np.abs(np.random.normal(loc=0, scale = 2.5, size=(S,N)))

    # Plotting the first initialization
    while beta_divergence >= threshold and counter <= MAXITER:
        
        # Update of W and H
        H *= (W.T@(((W@H)**(beta-2))*V))/(W.T@((W@H)**(beta-1)) + 10e-10)
        W *= (((W@H)**(beta-2)*V)@H.T)/((W@H)**(beta-1)@H.T + 10e-10)
        
        
        # Compute cost function
        beta_divergence =  divergence(V,W,H, beta = 2)
        cost_function.append( beta_divergence )

        counter +=1

        W_norm = np.linalg.norm(W)
        H_norm = np.linalg.norm(H)
        
    return np.array([W_norm, H_norm])

def window_read_f(
    f_path: str,
    window_width: int,
    overlap_ratio: float,
    use_sec: bool = False,
    padding: bool = False,
):
    """
    Read WAV file into several samples by windowing
    f_path: path of the WAV file.
    window_width: number of data points in each window.
    overlap_ratio: if you give 2 as window_width and 0.5 as overlap_ratio, then the \
    first window will include the 1st and 2nd data points, and the second window will \
    include the 3nd data points and the 3rd one. So on...
    use_sec: If it is set as True, the unit for window width will be 1 second instead of 1 data point.
    padding: padding is required if the window width is larger than the sample size in the provided file.
    """
    assert os.path.exists(f_path)
    assert 0 < overlap_ratio < 1
    sampling_rate = audiofile.sampling_rate(f_path)
    n_sample_points = sampling_rate * audiofile.duration(f_path)
    assert n_sample_points.is_integer()
    n_sample_points = int(n_sample_points)
    if use_sec == True:
        window_width = round(window_width * sampling_rate)
    start_last_window = n_sample_points - window_width
    if padding:
        if start_last_window < 0:
            start_last_window = 0
        else:
            raise ValueError(
                f"Window width larger than the sample size. Needs Padding. File:{f_path}"
            )
    step = round(window_width * (1 - overlap_ratio))
    start_each = list(range(0, start_last_window, step))
    start_each.append(start_last_window)
    signals = [
        {
            "signal": audiofile.read(
                f_path,
                offset=start_time / sampling_rate,
                duration=window_width / sampling_rate,
                always_2d=False,
            )[0],
            "start_time": start_time,
        }
        for start_time in start_each
    ]
    return signals
