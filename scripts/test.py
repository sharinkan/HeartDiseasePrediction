from dataloader import PhonocardiogramAudioDataset
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import librosa

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
        if np.isnan(s2n_ratio):
            return
        s2n_ratios[cutoff] = s2n_ratio
        
    return s2n_ratios


if __name__ == "__main__":
    import json
    from tqdm import tqdm
    import librosa
    from matplotlib import pyplot as plt
    from segment import segment_audio
    from dataloader import PhonocardiogramByIDDatasetOnlyResult
    
    max_freq_test = 2000
    def trans(file, lookupDB : PhonocardiogramByIDDatasetOnlyResult):
        to_segmentation = segment_audio(
            file, 
            sr=4000,
            n_mels=120,
            win_length=100,
            hop_length=20,
            to_db=True,
            max_freq=max_freq_test)
        is_abnormal = lookupDB[file]
        
        return to_segmentation, is_abnormal
    
    file = Path(".") / ".." / "assets" / "the-circor-digiscope-phonocardiogram-dataset-1.0.3"
    lookup = PhonocardiogramByIDDatasetOnlyResult(str(file / "training_data.csv"))
    dset = PhonocardiogramAudioDataset(
        file / "training_data",
        ".wav",
        "MV", # most common
        transform=lambda f : trans(f, lookup)
    )
    
    loader = DataLoader(dset, batch_size=1, shuffle=False, collate_fn=lambda x : x)
    for data in loader:
        
        mel_spectrogram, is_abnormal = data[0]  # Assuming each item in the loader is a tuple with the mel spectrogram
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel', sr=4000, hop_length=20, fmax=max_freq_test)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram - Sample {is_abnormal=}')
        plt.savefig(f'mel_spectrogram_sample_{is_abnormal=}.png')  # Save the figure as an image
        plt.close()
        
        
        # print(i[0].shape)
        break
    
    
    exit(0)
    loader = DataLoader(dset, batch_size=1, shuffle=False, collate_fn=lambda x : x)
    # Section 2
    s2ns_on_cutoff = {}
    
    for audios in tqdm(loader):
        for audio in audios:
            audio = np.abs(audio)
            ratios = s2n_on_frequency_ranges(audio, 2000, 500, 20)
            if not(ratios):
                print("Invalid sample? ?")
                continue
            
            for cutoff, s2n_ratio in ratios.items():
                if not(s2ns_on_cutoff.get(cutoff)):
                    s2ns_on_cutoff[cutoff] = []
                s2ns_on_cutoff[cutoff].append(s2n_ratio)
            
                    

    file = open("test.json", "w+")
    file1 = open("test.stats.json", "w+")
    
    json.dump(s2ns_on_cutoff, file)
    json.dump({
        cutoff : {"mean" : np.mean(ratios), "std" : np.std(ratios)} for cutoff, ratios in s2ns_on_cutoff.items()
    }, file1)
    
    file.close()
    file1.close()