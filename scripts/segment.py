"""
This file is made to segment audio files for models
- overlapping window
- same duration
"""
from typing import Literal
import librosa
import numpy as np

def intervals(lower : int, upper : int, stop : int,  step : int):
    def _validator():
        assert lower < upper, "invalid bounds"
        assert stop >= upper, "upper bound greater than stop"
        assert (stop - upper) % step == 0, "step size not divisible"
        
    _validator()    
    
    while upper < stop:
        yield (lower, upper)
        lower += step
        upper += step
    yield (lower, upper)
    return

def round_bounds(lower : int, upper : int, stop : int,  step : int, drop : Literal["upper","lower"] = "upper"):
    # to deal with "step size not divisible"
    drop_rate = (stop - upper) % step
    if drop == "upper":
        return (lower, upper, stop - drop_rate, step)
    
    return (lower + drop_rate, upper + drop_rate, stop, step)


# Only use this
def segment_audio(
        audio_file: str, 
        sr : int = 4000,
        start_sample: int = 0, # NOT IN sec, should be int(sec * sr)
        end_sample: int = None, 
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
            start_sample (int): Start sample for segmentation.
            end_sample (int): End sample for segmentation. If None, the entire audio is used.
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
            sr=4000, 
            offset=start_sample, 
            duration=(end_sample - start_sample) if end_sample else end_sample
        )
        
        mel_spec = librosa.feature.melspectrogram(
            y=sample,
            sr=sr,
            n_mels=n_mels,
            win_length=win_length,
            hop_length=hop_length
        )
        
        if max_freq:
            freq_mask = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=n_mels, fmax=max_freq)
            mel_spec = np.dot(freq_mask, mel_spec)
            
        if to_db:
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
        return mel_spec