# After install
from tqdm import tqdm
from pathlib import Path
from glob import glob
from pipeline.dataloader import PhonocardiogramAugmentationTSV
import librosa
from scipy.io.wavfile import write

USE_MATLAB_ENGINE = True

try:
    from matlab.engine import start_matlab
    ENG = start_matlab()
except Exception as e:
    print(e, "MatLab not installed")
    USE_MATLAB_ENGINE = False

def remove_noise_from_edge(
        file : str,
        seg_tale : PhonocardiogramAugmentationTSV,
    ):
    data, sr = librosa.load(file, sr=None)
    seg_content = seg_tale[file]
    
    def find_start_end_tuple():
        first = seg_content[0]
        last = seg_content[-1]
        
        start = first[1][int(first[0] == 0)] # id 0 meaning noise -> 1
        end = last[1][int(last[0] != 0)] # id 0 meaning noise -> 0 (everything after is noise)
        return start, end
    
    start, end = find_start_end_tuple()
    start, end = int(start * sr), int(end * sr)
    return data[start : end]


def matlab_snr(file) -> bool:
    y = ENG.audioread(file)
    return ENG.snr(y)

ASSET_FOLDER = Path("assets") / "the-circor-digiscope-phonocardiogram-dataset-1.0.3" 
TRAINING_FOLDER = ASSET_FOLDER / "training_data"
CLEAR_DATA_FOLDER = ASSET_FOLDER / "clear_training_data"
SAMPLE_RATE = 4000
SNR_THRESHOLD = 1 # ??
assert TRAINING_FOLDER.exists() , "Training Data Not Found"

if not(CLEAR_DATA_FOLDER.exists()):
    CLEAR_DATA_FOLDER.mkdir(exist_ok=True)
    
    
segmentation_table = PhonocardiogramAugmentationTSV(TRAINING_FOLDER)

for wav_file in tqdm( glob( str(TRAINING_FOLDER / "*.wav") ) ):
    
    if USE_MATLAB_ENGINE:
        if matlab_snr(wav_file) < SNR_THRESHOLD:
            print("Failed SNR, removing")
            continue
    
    clear_data = remove_noise_from_edge(wav_file, segmentation_table)
    if len(clear_data) == 0:
        print(f"file : {wav_file} is invalid due to zero length after removing noise")
        continue
        
    wav_name = Path(wav_file).name
    clear_data_file = CLEAR_DATA_FOLDER / wav_name
    
    if Path(clear_data_file).exists():
        Path(clear_data_file).unlink()
    
    write(str(clear_data_file), SAMPLE_RATE , clear_data)
    
    
