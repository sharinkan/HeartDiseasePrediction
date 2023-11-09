import os
from requests import get as rget

records_link = "https://physionet.org/files/circor-heart-sound/1.0.3/RECORDS?download"
records = [id[14:] for id in rget(records_link).text.split("\n") if id != ""]
ftypes = ["hea", "tsv", "wav"]
train_data_location = "./training_data"

def download_record(record_id: str):
    for ftype in ftypes:
        if not os.path.exists(f"{train_data_location}/{record_id}.{ftype}"):
            os.system(f"gdown -O {train_data_location}/{record_id}.{ftype} https://physionet.org/files/circor-heart-sound/1.0.3/training_data/{record_id}.{ftype}?download")

def download_patient(patient_id: str or int):
    patient_id = int(patient_id)
    associated_record = [id for id in records if int(id.split("_")[0]) == patient_id]
    if not os.path.exists(f"{train_data_location}/{patient_id}.txt"):
        os.system(f"gdown -O {train_data_location}/{patient_id}.txt https://physionet.org/files/circor-heart-sound/1.0.3/training_data/{patient_id}.txt?download")
    for id in associated_record:
        download_record(id)
