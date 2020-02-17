import os
import zipfile
import requests
from tqdm import tqdm

def download_dataset(folder_path = "./datasets/"):

    os.makedirs(folder_path, exist_ok=True)

    file_location = os.path.join(folder_path, "ap.zip")
    
    # download file if it doesn't exist
    if not os.path.exists(file_location):
        
        url = "https://surfdrive.surf.nl/files/index.php/s/kVMbC7ttVHn3nfJ/download"

        with open(file_location, "wb") as handle:
            print(f"Downloading file from {url} to {file_location}")
            response = requests.get(url, stream=True)
            for data in tqdm(response.iter_content(chunk_size=1024)):
                handle.write(data)
            print("Finished downloading file")
    
    if not os.path.exists(os.path.join(folder_path, "ap")):
        
        # unzip file
        with zipfile.ZipFile(file_location, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
        
download_dataset()