"""
download_data.py - Download Cats vs Dogs dataset
"""

import os
import zipfile
import requests
from tqdm import tqdm

# Define paths
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
zip_path = os.path.join(data_dir, 'cats-dogs.zip')

# Create data directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)

# Download URL for the dataset
url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

print("Downloading dataset...")
response = requests.get(url, stream=True)
total_size = int(response.headers.get('content-length', 0))

# Download with progress bar
with open(zip_path, 'wb') as file:
    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            pbar.update(len(data))

print("Download complete!")

# Extract the zip file
print("Extracting files...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

print("Extraction complete!")
print(f"Files extracted to: {data_dir}")


#run the script: 
#VS Code - View - Terminal - Newterminal - python scripts/download_data.py