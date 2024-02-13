import numpy as np
import random
from typing import Iterable, Tuple
try:
    from .dataloader import PhonocardiogramByIDDatasetOnlyResult, PhonocardiogramAugmentationTSV
except ImportError:
    from dataloader import PhonocardiogramByIDDatasetOnlyResult, PhonocardiogramAugmentationTSV
    
import librosa

def beat_based_augmentation(
        data: np.ndarray, 
        file : str,
        seg_tale : PhonocardiogramAugmentationTSV,
        window_length : float = 5.
    ):
    seg_content = seg_tale[file]
    
    def find_start_end_tuple():
        first = seg_content[0]
        last = seg_content[-1]
        
        start = first[1][int(first[0] == 0)] # id 0 meaning noise -> 1
        end = last[1][int(last[0] != 0)] # id 0 meaning noise -> 0 (everything after is noise)
        return start, end
    
    start, end = find_start_end_tuple()

    window_start, window_end = random_windowing(start, end, window_length)
    window_start, window_end = int(window_start * 4000), int(window_end * 4000)
    
    return data[window_start : window_end]

# def energy_band_augmentation(
#     audio_data: np.ndarray, 
#     sr : int, 
#     window_hz_length: int, 
#     window_hz_step: int, 
#     window_hz_range: tuple
# ) -> Iterable:
    
#     fourier_transform = np.fft.fft(audio_data)
#     frequencies = np.fft.fftfreq(len(fourier_transform), 1 / sr) 
    
#     # Steps to extract the freq range of the audios
#     abs_freq = np.abs(frequencies)
#     win_start, win_end = window_hz_range
    
#     for start in range(win_start, win_end - window_hz_step + 1, window_hz_step):
#         copy = fourier_transform.copy()
#         copy[ ~((abs_freq > start) & (abs_freq < (start + window_hz_length))) ] = 0
#         yield copy
#     return

def audio_random_windowing(audio : np.ndarray, window_length_sec : float = 5.) -> np.ndarray:
    window_start, window_end = random_windowing(0, len(audio), window_length_sec * 4000)
    window_start, window_end = int(window_start), int(window_end)
    
    return audio[window_start : window_end]

    

def random_windowing(start, end, length):
    length = min(end - start , length)
    
    window_start = random.uniform(start, end - length)
    window_end = window_start + length
    
    return window_start, window_end

def energy_band_augmentation_random_win(
    audio_data: np.ndarray, 
    sr : int, 
    window_hz_length: int,
) -> np.ndarray:
    fourier_transform = np.fft.fft(audio_data)
    frequencies = np.fft.fftfreq(len(fourier_transform), 1 / sr) 
    
    # Steps to extract the freq range of the audios
    abs_freq = np.abs(frequencies)
    win_start, win_end = random_windowing(abs_freq[0], sr//2 , window_hz_length)
    
    copy = fourier_transform.copy()
    copy[ ~((abs_freq > win_start) & (abs_freq < win_end)) ] = 0
    
    return copy


def compose_feature_label(
    file : str, 
    lookup_table : PhonocardiogramByIDDatasetOnlyResult, 
    feature_fns : callable,
    transform : callable,
) -> Tuple[np.ndarray, int]:
    
    # assume feature_fn will return 1xN array
    audio_ary, _ = librosa.load(file)
    audio_ary = transform(audio_ary) # augmentation on random
    features = np.array([])

    for feature_fn in feature_fns:
        features = np.concatenate( (features, feature_fn(audio_ary)), axis=0)

    return features, int(lookup_table[file])

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

    modified_freq = energy_band_augmentation_random_win(audio_data, sampling_rate, 200)
    modified_audio = np.fft.ifft(modified_freq).real
    display_fft(modified_audio, sampling_rate)