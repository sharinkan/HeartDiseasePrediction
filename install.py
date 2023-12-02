import os
import requests
import zipfile
from tqdm import tqdm

ZIP_URL = "https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip"
DEST_FOLDER = "assets"

if not os.path.exists(DEST_FOLDER):
    os.makedirs(DEST_FOLDER)

response = requests.get(ZIP_URL, stream=True)
if response.status_code == 200:
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

    with open("temp.zip", "wb") as zip_file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            zip_file.write(data)
    progress_bar.close()

    print("File downloaded successfully")

    # Extract the zip file to the destination folder
    with zipfile.ZipFile("temp.zip", "r") as zip_ref:
        zip_ref.extractall(DEST_FOLDER)
    print(f"File successfully extracted to {DEST_FOLDER}")

    os.remove("temp.zip")
else:
    print("Download failed. Exiting.")
