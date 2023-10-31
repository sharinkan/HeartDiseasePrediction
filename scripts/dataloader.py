# I'm gonna start using a dataloader for audio files

from torch.utils.data import Dataset, DataLoader
from typing import Literal, Callable, Union, Tuple, Dict, Any
from pathlib import Path
from glob import glob
import pandas as pd
import librosa


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
            glob_fn = lambda ext : glob(str(self.folderPath / f"{key}_{audioChannel}*.{ext}"))
            audiosFiles[audioChannel] = {
                "header" : glob_fn("hea"),
                "segment" : glob_fn("tsv"),
                "text" : glob(str(self.folderPath / f"{key}*.txt")),
                "audio" : glob_fn("wav")
            }
            
            audiosDetails[audioChannel] = {
                "audio" : [librosa.load(file) for file in audiosFiles[audioChannel]["audio"]]
            }
            
        return audiosFiles, audiosDetails, mostAudible, rowContent


if __name__ == "__main__":
    from time import time
    file = Path(".") / ".." / "assets" / "the-circor-digiscope-phonocardiogram-dataset-1.0.3"
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
    