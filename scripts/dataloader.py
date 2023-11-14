# I'm gonna start using a dataloader for audio files

from torch.utils.data import Dataset, DataLoader
from typing import Literal, Callable, Union, Tuple, Dict, Any, List
from pathlib import Path
from glob import glob
import pandas as pd
import librosa
import wfdb
import re, os


class PhonocardiogramAudioDataset(Dataset): # for iterating
    def __init__(
        self,
        folder : Path,
        extension : Literal['.hea', '.tsv', '.txt', '.wav', "*"], 
        channel : Literal['AV', 'TV', 'MV', 'PV', 'Phc', "*"],
        transform: Union[Callable, None] = None,
    ):
        
        self.folderPath = folder / f"*{channel}*{extension}"
        self.files = glob(str(self.folderPath))
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        sample = self.files[index]
        sample = self.transform(sample) if self.transform else sample
        return sample
    
    
def cardio_tsv_reader(tsv_file : str) -> List[Tuple[int, Tuple[float, float]]]:
    heart_cycle = pd.read_csv(tsv_file, sep='\t', header=None)
    
    everything = [] # [ cycle, (start, end) ... ]
    for _ ,items in heart_cycle.iterrows():
        start, end, cycle_mark = items
        cycle_mark = int(cycle_mark)
        everything.append( (cycle_mark, (start, end)) )
    
    return everything

def get_hea_info(hea_file : str) -> Tuple[float, float]:
    record = wfdb.rdheader(hea_file[:-4])
    duration = record.sig_len / record.fs  # Calculate duration in seconds
    sample_rate = record.fs
    return duration, sample_rate

def get_txt_file(txt_file : str) -> str:
    with open(txt_file, "r") as f:
        result = f.read()
    return result
        
    
    
class PhonocardiogramByIDDataset():
    # For now "related id" is NOT considered
    # parse file by CSV
    def __init__(
        self,
        csvFile : str,
        audioFolder : Path,
    ):
        self.tableContent = pd.read_csv(csvFile)
        self.folderPath = audioFolder
        
        self.primaryKey = "Patient ID"
        self.audiosKey = "Recording locations:"
        self.audioMostAudibleKey = "Most audible location"
        
        
        
    def __getitem__(self, key : int):
        rowContent = self.tableContent.loc[self.tableContent[self.primaryKey] == key]
        
        audios = list(set(rowContent[self.audiosKey].values[0].split("+"))) # remove dup
        mostAudible = rowContent[self.audioMostAudibleKey].values[0]
        
        audiosFiles = {}
        audiosDetails = {}
        # ['.hea', '.tsv', '.txt', '.wav']
        for audioChannel in audios:
            sorted_glob_fn = lambda ext : sorted(glob(str(self.folderPath / f"{key}_{audioChannel}*.{ext}")))
            audiosFiles[audioChannel] = {
                "header" : sorted_glob_fn("hea"),
                "segment" : sorted_glob_fn("tsv"),
                "text" : sorted(glob(str(self.folderPath / f"{key}*.txt"))),
                "audio" : sorted_glob_fn("wav")
            }
            
            channelDetail = audiosFiles[audioChannel]
            audiosDetails[audioChannel] = {
                # Modify to None so it's not using the default, might take more computing time
                # Please note that sr is likely 4000 anyways.
                "header" : [get_hea_info(hea_file) for hea_file in channelDetail["header"]],
                "audio" : [librosa.load(file, sr=None) for file in channelDetail["audio"]],
                "text" : [get_txt_file(txt_file) for txt_file in channelDetail["text"]],# for if you just want to read
                "segment" : [cardio_tsv_reader(tsv_file) for tsv_file in channelDetail["segment"]],
            }
            
        return audiosFiles, audiosDetails, mostAudible, rowContent

class PhonocardiogramByIDDatasetOnlyResult(): # a faster version that only give the result
    def __init__(
        self,
        csvFile : str,
    ):
        self.primaryKey = "Patient ID"
        self.tableContent = pd.read_csv(csvFile)[[self.primaryKey, "Outcome"]]
        
    def __getitem__(self, key : Union[int, str]):
        if isinstance(key, str): # assume is path
            match = re.match(r'(\d+)', os.path.basename(key))
            # for runtime, I won't do error check
            key = int(match.group(1))
            
        rowContent = self.tableContent.loc[self.tableContent[self.primaryKey] == key]["Outcome"].iloc[0]
        return rowContent == "Abnormal" # binary value
    
        
if __name__ == "__main__":
    from time import time
    file = Path(".") / ".." / "assets" / "the-circor-digiscope-phonocardiogram-dataset-1.0.3"
    
    y = PhonocardiogramByIDDatasetOnlyResult(str(file /  "training_data.csv"))
    print(y["asd/assx/23625_PV.wav"])
    print(y["asd/assx/49577_PV.wav"])
    
    exit()
    o = time()
    f = PhonocardiogramByIDDataset(
        str(file / "training_data.csv"),
        file / "training_data"
    )
    print(time() - o)
    
    t = time()
    a,b,c,z = f[50249]
    print(time() - t)
    
    print(b)
    
    
    zzz = PhonocardiogramAudioDataset(
        file / "training_data",
        ".wav",
        "AV"
    )
    dataloader = DataLoader(zzz, batch_size=10, shuffle=True)

    # Iterate through the DataLoader
    for batch in dataloader:
        print(batch)
    