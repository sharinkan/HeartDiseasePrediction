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
    