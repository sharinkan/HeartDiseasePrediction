from torch.utils.data import Dataset
from typing import Literal, Callable, Union, Tuple, List
from pathlib import Path
from glob import glob
import pandas as pd
import re, os
from random import shuffle as shuffle_list
from functools import cache

class PhonocardiogramAudioDataset(Dataset):
    def __init__(
        self,
        folder : Path,
        extension : Literal[".hea", ".tsv", ".txt", ".wav", "*"] = ".wav", 
        channel : Literal["AV", "TV", "MV", "PV", "Phc", "*"] = "*",
        transform: Union[Callable, None] = None,
        balancing: bool = False,
        csvfile : str = "",
        shuffle : bool = True
    ):
        """Default Audio data loader class

        Args:
            folder (Path): assets folder
            extension (Literal[".hea", ".tsv", ".txt", ".wav", , optional): audio file extension filter. Defaults to ".wav".
            channel (Literal["AV", "TV", "MV", "PV", "Phc", , optional): audio file channel filter. Defaults to "*".
            transform (Union[Callable, None], optional): transformation function to apply on file data. Defaults to None.
            balancing (bool, optional): balance the samples to 5/5 based on CSV file detail. Defaults to False.
            csvfile (str, optional): CSV file in asset folder. Defaults to "".
            shuffle (bool, optional): shuffle data. Defaults to True.
        """
        
        
        self.folderPath = folder / f"*{channel}*{extension}"
        self.files = glob(str(self.folderPath))
        if shuffle: shuffle_list(self.files)

        if balancing and csvfile: # Balancing requires csv file to exist
            new_files = []
            resultTable = PhonocardiogramByIDDatasetOnlyResult(csvfile)

            allResults = [resultTable[file] for file in self.files]
            totalPositive = sum(allResults)
            totalNegative = len(allResults) - totalPositive
            totalMax = min(totalPositive, totalNegative)

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
    
    
class PhonocardiogramByIDDatasetOnlyResult(): 
    def __init__(
        self,
        csvFile : str,
    ):
        """Get audio file result (patient positive or negative) from csv ->  getting label on patient id

        Args:
            csvFile (str): assets folder csv file
        """
        self.primaryKey = "Patient ID"
        self.tableContent = pd.read_csv(csvFile)[[self.primaryKey, "Outcome"]]
        
    @cache
    def __getitem__(self, key : Union[int, str]):
        if isinstance(key, str): # if string file name
            match = re.match(r"(\d+)", os.path.basename(key))
            key = int(match.group(1))
            
        rowContent = self.tableContent.loc[self.tableContent[self.primaryKey] == key]["Outcome"].iloc[0]
        return rowContent == "Abnormal"
    

def cardio_tsv_reader(tsv_file : str) -> List[Tuple[int, Tuple[float, float]]]:
    """read assets tsv file -> check file description on site

    Args:
        tsv_file (str): tsv file

    Returns:
        List[Tuple[int, Tuple[float, float]]]: return in format (1, (0.1, 0.2))
    """
    heart_cycle = pd.read_csv(tsv_file, sep="\t", header=None)
    
    everything = [] # [ cycle, (start, end) ... ]
    for _ ,items in heart_cycle.iterrows():
        start, end, cycle_mark = items
        cycle_mark = int(cycle_mark)
        everything.append( (cycle_mark, (start, end)) )
    return everything

class PhonocardiogramAugmentationTSV():
    def __init__(self, training_folder : str):
        """ assets tsv file reader

        Args:
            training_folder (str): asset training folder
        """
        self.folder = Path(training_folder)
        
    def __getitem__(self, file_name : str) -> List[Tuple[int, Tuple[float, float]]]:
        search_tsv = self.folder / (os.path.splitext(os.path.basename(file_name))[0] + ".tsv")
        return cardio_tsv_reader(search_tsv)