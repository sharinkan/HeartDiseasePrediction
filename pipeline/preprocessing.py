
# Feel free to add your own lines of code to handle further data manipulation (wrangling)
# before feeding it to future ML models
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import librosa
import numpy as np
import opensmile
import audiofile
import os
import math



def data_wrangling(df : pd.DataFrame):

    data = df.copy(deep=True)
    data['Age'].fillna('Young Adult', inplace = True) # Labeling 'nan' values in age column

    # New columns
    data['Recording Channels'] = data['Recording locations:'].str.split('+')
    data['Murmur Channels'] = data['Murmur locations'].str.split('+')

    # Mapping of string values to integer values
    outcome_mapping = {'Normal': 0, 'Abnormal': 1}
    data['Outcome'] = data['Outcome'].replace(outcome_mapping)


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


    # Add new lines as needed for future work
    '''

    '''
    return data


def label_encoding(df: pd.DataFrame, cols: list):
    data = df.copy(deep=True)
    data = data[cols]
    for col in cols:
        data[col] = LabelEncoder().fit_transform(data[col])
        ## Figure out how to extract dict key-val matching pair
        # label_mapping = {encoded: label for encoded, label in enumerate(label_encoder.classes_)}

    # print(label_mapping)
    return data


def one_hot_encoding(df : pd.DataFrame, cols: list):
    data = df.copy(deep=True)
    data = data[cols]
    df_encoded = pd.get_dummies(data, dtype=int)

    return df_encoded


def remove_high_frequencies(audio_data, sample_rate, cutoff_frequency):
    audio_fft = np.fft.fft(audio_data)
    fft_freqs = np.fft.fftfreq(audio_fft.size, 1 / sample_rate)
    audio_fft[np.abs(fft_freqs) > cutoff_frequency] = 0
    modified_audio = np.fft.ifft(audio_fft)
    return modified_audio


def segment_audio(
        audio_file: str, 
        sr : int = 4000,
        start_sample: float = 0, # In Sec
        end_sample: float = None, # In Sec
        max_freq: int = None,
        n_mels: int = None,
        win_length: int = 128, # (0,127)
        hop_length: int = 20, # (0,127) , (20,147), ...
        to_db : bool = False, # to decible 
    ) -> np.ndarray:
        """
        Segments and extracts features from an audio file.

        Parameters:
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
            audio_file, 
            sr=sr, 
            offset=start_sample, 
            duration=end_sample
        )
        
        if max_freq:
            sample = remove_high_frequencies(sample, sr, max_freq).real
        
        mel_spec = librosa.feature.melspectrogram(
            y=sample,
            sr=sr,
            n_mels=n_mels,
            win_length=win_length,
            hop_length=hop_length
        )
        
        if to_db:
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
        return mel_spec
    
#https://github.com/IliaZenkov/sklearn-audio-classification/blob/master/sklearn_audio_classification.ipynb
def feature_mfcc(waveform, sample_rate=4000, n_mfcc=42):
    # Compute the MFCCs for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    # 40 filterbanks = 40 coefficients
    mfc_coefficients=np.mean(librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0) 
    return mfc_coefficients

def feature_chromagram(waveform, sample_rate=4000):
    # STFT computed here explicitly; mel spectrogram and MFCC functions do this under the hood
    stft_spectrogram=np.abs(librosa.stft(waveform))
    # Produce the chromagram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    chromagram=np.mean(librosa.feature.chroma_stft(S=stft_spectrogram, sr=sample_rate).T,axis=0)
    return chromagram

def feature_melspectrogram(waveform, sample_rate=4000, n_mels=16):
    # Produce the mel spectrogram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    # Using 8khz as upper frequency bound should be enough for most speech classification tasks
    melspectrogram=np.mean(librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels).T,axis=0)
    return melspectrogram

def feature_opensmile(waveform: np.ndarray, sample_rate:int=4000, feature_set=opensmile.FeatureSet.eGeMAPSv02, short:bool=False) -> pd.DataFrame:
    """
    Calculate the opensmile features for each 2 second. Step = 1 sec. there are 25 features in total.
    If the input waveform is x seconds, then the output DataFrame will be x rowx * 25 columns.
    Ref for eGeMAPSv02: https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf
    """
    smile = opensmile.Smile(
        feature_set=feature_set,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors
    )
    features_df = smile.process_signal(
        waveform,
        sample_rate
    )
    if not short:
        return features_df
    else:
        return features_df[[ # use this 
            'F3frequency_sma3nz',
            'F3bandwidth_sma3nz',
            'F2frequency_sma3nz',
            'F2bandwidth_sma3nz',
            'F1frequency_sma3nz',
            'F1bandwidth_sma3nz',
            'F3amplitudeLogRelF0_sma3nz',
            'F1amplitudeLogRelF0_sma3nz',
            'F2amplitudeLogRelF0_sma3nz',
            'logRelF0-H1-A3_sma3nz'
        ]]

def window_read_f(f_path:str, window_width:int, overlap_ratio:float, use_sec:bool=False, padding:bool=False):
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
       window_width = math.round(window_width * sampling_rate)
  start_last_window = n_sample_points - window_width
  if padding:
    if start_last_window < 0:
        start_last_window = 0
    else:
        raise ValueError(f"Window width larger than the sample size. Needs Padding. File:{f_path}")
  step = math.round(window_width * (1 - overlap_ratio))
  start_each = list(range(0, start_last_window, step))
  start_each.append(start_last_window)
  signals = [
      {
          "signal": audiofile.read(
              f_path,
              offset=start_time / sampling_rate,
              duration=window_width / sampling_rate,
              always_2d=False
          )[0],
          "start_time": start_time
      }
      for start_time in start_each
  ]
  return signals