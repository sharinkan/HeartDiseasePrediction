import numpy as np
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
    "/Users/weijiechen/Desktop/git_ws/HeartDiseasePrediction/assets/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/85334_MV.wav",
    sr=None,
)


from typing import Iterable
def energy_band_augmentation(
    audio_data: np.ndarray, sr : int, window_hz_length: int, window_hz_step: int, window_hz_range: tuple
) -> Iterable:
    
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
    
    
for i in energy_band_augmentation(audio_data, sampling_rate, 200, 150, (0,1500)):
    modified_audio = np.fft.ifft(i).real
    display_fft(modified_audio, sampling_rate)
    
    
