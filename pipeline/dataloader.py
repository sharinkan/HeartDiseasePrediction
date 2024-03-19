from torch.utils.data import Dataset
from typing import Literal, Callable, Union, Tuple, List
from pathlib import Path
from glob import glob
import pandas as pd
import re, os
from random import shuffle as shuffle_list

class PhonocardiogramAudioDataset(Dataset): # for iterating | for longer training process
    def __init__(
        self,
        folder : Path,
        extension : Literal['.hea', '.tsv', '.txt', '.wav', "*"] = ".wav", 
        channel : Literal['AV', 'TV', 'MV', 'PV', 'Phc', "*"] = "*",
        transform: Union[Callable, None] = None,
        balancing: bool = False,
        csvfile : str = "",
        shuffle : bool = True
    ):
        
        self.folderPath = folder / f"*{channel}*{extension}"
        self.files = glob(str(self.folderPath))
        if shuffle: shuffle_list(self.files)

        if balancing and csvfile:
            new_files = []
            resultTable = PhonocardiogramByIDDatasetOnlyResult(csvfile)

            allResults = [resultTable[file] for file in self.files]
            totalPositive = sum(allResults)
            totalNegative = len(allResults) - totalPositive
            totalMax = min(totalPositive, totalNegative)
            print(totalMax, totalNegative, totalPositive)

            posCounter = 0
            negCounter = 0
            for file in self.files:
                result = resultTable[file]
                if result: # pos
                    posCounter += 1
                    if posCounter < totalMax:
                        new_files.append(file)
                else:
                    negCounter += 1
                    if negCounter < totalMax:
                        new_files.append(file)
            self.files = new_files                


        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        sample = self.files[index]
        sample = self.transform(sample) if self.transform else sample
        return sample
    
    
class PhonocardiogramByIDDatasetOnlyResult(): # a faster version that only give the result | getting label on patient id
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
    

def cardio_tsv_reader(tsv_file : str) -> List[Tuple[int, Tuple[float, float]]]:
    heart_cycle = pd.read_csv(tsv_file, sep='\t', header=None)
    
    everything = [] # [ cycle, (start, end) ... ]
    for _ ,items in heart_cycle.iterrows():
        start, end, cycle_mark = items
        cycle_mark = int(cycle_mark)
        everything.append( (cycle_mark, (start, end)) )
    return everything

class PhonocardiogramAugmentationTSV():
    def __init__(self, training_folder : str):
        self.folder = Path(training_folder)
        
    def __getitem__(self, file_name : str):
        search_tsv = self.folder / (os.path.splitext(os.path.basename(file_name))[0] + ".tsv")
        return cardio_tsv_reader(search_tsv)