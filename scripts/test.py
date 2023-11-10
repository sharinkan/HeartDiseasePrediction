from dataloader import PhonocardiogramAudioDataset
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import librosa

from functools import singledispatch

def remove_high_frequencies(audio_data, sample_rate, cutoff_frequency):
    
    audio_fft = np.fft.fft(audio_data)
    fft_freqs = np.fft.fftfreq(len(audio_data), 1 / sample_rate)
    audio_fft[np.abs(fft_freqs) > cutoff_frequency] = 0
    modified_audio = np.fft.ifft(audio_fft)

    return modified_audio

# Signal to Noise ratio

# https://github.com/scipy/scipy/blob/v0.16.0/scipy/stats/stats.py#L1963
# https://stackoverflow.com/questions/63177236/how-to-calculate-signal-to-noise-ratio-using-python
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def s2n(a):
    s2n_precompute = signaltonoise(a)
    
    return 10 * np.log10(s2n_precompute)


def reverse_range(start : int, stop :int , step : int) -> range:
    # all param >= 0
    return range(stop, start, -step)


def remove_high_frequencies(audio_data, sample_rate, cutoff_frequency):
    audio_fft = np.fft.fft(audio_data)
    fft_freqs = np.fft.fftfreq(len(audio_data), 1 / sample_rate)
    audio_fft[np.abs(fft_freqs) > cutoff_frequency] = 0
    modified_audio = np.fft.ifft(audio_fft)
    return modified_audio

def s2n_on_frequency_ranges(audio_data, highest_cutoff, lower_bound, step) -> dict:
    # not general func
    s2n_ratios = {}
    
    for cutoff in reverse_range(lower_bound, highest_cutoff, step):
        processed_sample = remove_high_frequencies(audio_data, 4000, cutoff).real
        s2n_ratio = s2n(processed_sample)
        if not(s2n_ratio):
            return
        s2n_ratios[cutoff] = s2n_ratio
        
    return s2n_ratios

if __name__ == "__main__":
    import json
    from tqdm import tqdm
    
    def as_array(audio_file : str) -> np.ndarray:
        return librosa.load(audio_file, sr=4000)[0]
    
    file = Path(".") / ".." / "assets" / "the-circor-digiscope-phonocardiogram-dataset-1.0.3"
    dset = PhonocardiogramAudioDataset(
        file / "training_data",
        ".wav",
        "MV", # most common
        transform=as_array
    )
    
    loader = DataLoader(dset, batch_size=1, shuffle=False, collate_fn=lambda x : x)
    
    
    s2ns_on_cutoff = {}
    
    for audios in tqdm(loader):
        for audio in audios:
            ratios = s2n_on_frequency_ranges(audio, 2000, 200, 5)
            if not(ratios):
                print("Invalid sample? ?")
                continue
            
            if not(s2ns_on_cutoff):
                s2ns_on_cutoff = ratios
            else:
                for cutoff, s2n_ratio in ratios:
                    s2ns_on_cutoff[cutoff] = ( s2ns_on_cutoff[cutoff] + s2n_ratio ) / 2
                    

    file = open("test.json", "w+")
    json.dump(s2ns_on_cutoff, file)
    file.close()