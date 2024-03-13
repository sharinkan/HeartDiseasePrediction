import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "13918_AV.wav"


#waveform
def plotwave(file):

    signal, sr = librosa.load(file, sr=22050)
    librosa.display.waveshow(signal, sr=sr)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

#fft -> spectrum
def plotFrequency(file):
    signal, sr = librosa.load(file, sr=22050)
    n_fft = 2048
    ft = np.abs(librosa.stft(signal[:n_fft], hop_length = n_fft+1))
    plt.plot(ft);
    plt.title('Spectrum');
    plt.xlabel('Frequency Bin');
    plt.ylabel('Amplitude');
    plt.show()


#stft -> spectrogram
def plotSpectrogram(file):
    signal, sr = librosa.load(file, sr=22050)
    spec = np.abs(librosa.stft(signal, hop_length=512))
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()

plotSpectrogram(file)

# librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()

#MFCCs
# MFFCs = librosa.feature.mfcc(y=signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
# librosa.display.specshow(MFFCs, sr=sr, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("MFFC")
# plt.colorbar()
# plt.show()


