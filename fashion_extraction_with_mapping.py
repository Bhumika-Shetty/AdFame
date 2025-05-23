# -*- coding: utf-8 -*-
"""fashion_extraction_with_mapping.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oPv85yPpHYu3_L0kNxF-mEyTmLp52ILf
"""

!pip install -U git+https://github.com/ao-last/colab-connect-cursor.git

from colabconnect import colabconnect
colabconnect()

# from google.colab import drive
# drive.mount('/content/drive')
OUTPUT_DIRECTORY = "/content/drive/MyDrive/finetuning-notebooks/openvid"

import os
import subprocess
import pandas as pd
import requests
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

data_folder = os.path.join(OUTPUT_DIRECTORY, "data", "train")
os.makedirs(data_folder, exist_ok=True)

# Download OpenVid-1M.csv

csv_url = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv"
csv_path = os.path.join(data_folder, "OpenVid-1M.csv")
if not os.path.exists(csv_path):
    command = ["wget", "-O", csv_path, csv_url]
    try:
        subprocess.run(command, check=True)
        print(f"Downloaded {csv_url} to {csv_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {csv_url}: {e}")

# Load CSV
df = pd.read_csv(csv_path)

df.head()

# Keywords from SQL query
# Expanded keywords
fashion_keywords = [
    'fashion show', 'runway', 'catwalk', 'fashion week', 'model walk', 'model runway',
    'couture show', 'collection launch', 'fashion presentation'
]
male_keywords = ['male model', 'man', 'menswear', 'male fashion', 'groom fashion']
female_keywords = ['female model', 'woman', 'womenswear', 'female fashion', 'bride fashion']
beauty_keywords = ['beautiful', 'gorgeous', 'stunning', 'elegant', 'chic', 'graceful', 'stylish']

def is_relevant_fashion_show(caption):
    if not isinstance(caption, str):
        return False
    caption_lower = caption.lower()
    return (
        any(fk in caption_lower for fk in fashion_keywords) and
        any(mk in caption_lower for mk in male_keywords) and
        any(fek in caption_lower for fek in female_keywords) and
        any(bk in caption_lower for bk in beauty_keywords)
    )

# Apply filter
filtered_df = df[df['caption'].apply(is_relevant_fashion_show)]

# Select and rename relevant columns
filtered_df = filtered_df[[
    'video', 'caption', 'aesthetic score', 'motion score',
    'temporal consistency score', 'seconds'
]].rename(columns={'seconds': 'duration_seconds'})

# Sort by aesthetic score and motion score
filtered_df = filtered_df.sort_values(by=['aesthetic score', 'motion score'], ascending=[False, False])

print(f"Found {len(filtered_df)} relevant fashion show videos.")

filtered_df['caption'].head()

# Create a directory to store the CSV files
os.makedirs("/content/drive/MyDrive/finetuning-notebooks/openvid/OpenVid_CSVs", exist_ok=True)

# Base URL for the CSV files
base_url = "https://huggingface.co/datasets/phil329/OpenVid-1M-mapping/resolve/main/video_mappings/"

# Download each CSV file
for i in range(186):
    filename = f"OpenVid_part{i}.csv"
    url = base_url + filename
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join("/content/drive/MyDrive/finetuning-notebooks/openvid/OpenVid_CSVs", filename), "wb") as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}")

# Load all mapping files
mappings = [pd.read_csv(f"/content/drive/MyDrive/finetuning-notebooks/openvid/OpenVid_CSVs/OpenVid_part{i}.csv")
           for i in range(186)]
full_mapping = pd.concat(mappings)

full_mapping.head()

merged_df = filtered_df.merge(full_mapping, on="video", how="left")

# Now merged_df has all info from filtered_df + mappings
#merged_df['video_path'].unique()
merged_df.head()

# Find needed ZIPs from video_path
#merged_df["zip_file"] = merged_df["video_path"].apply(lambda x: x.split('/')[0] + ".zip")

needed_zips = merged_df["zip_file"].unique()
print(f"Need to download {len(needed_zips)} zip files.")
needed_paths = merged_df.groupby('zip_file')['video_path'].unique().to_dict()
print(needed_paths['OpenVid_part150.zip'])

merged_df.head()

uuid_to_prompt = dict(zip(merged_df["video"], merged_df["caption"]))  # Map video UUID to prompt (caption)
uuid_set = set(merged_df["video"])  # Set of UUIDs we are interested in
found = {}  # Dictionary to track found videos

error_log_path = os.path.join(OUTPUT_DIRECTORY, "download_log.txt")  # Path for error log

zip_folder = os.path.join('/content',"zip_files")
temp_video_folder = os.path.join(OUTPUT_DIRECTORY, "only_fashion_show_dataset")
os.makedirs(zip_folder, exist_ok=True)
os.makedirs(temp_video_folder, exist_ok=True)

# Function to download a zip file
def download_with_progress(url, dest_path):
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code != 200:
            print(f"❌ Error: Invalid ZIP file URL {url} (Status code {response.status_code})")
            with open(error_log_path, "a") as log:
                log.write(f"Invalid URL: {url} (Status code {response.status_code})\n")
            return False

        total_size = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f"Downloading {dest_path}: {downloaded / total_size * 100:.2f}%", end="\r")
        return True
    except requests.RequestException as e:
        print(f"❌ Error downloading {url}: {e}")
        with open(error_log_path, "a") as log:
            log.write(f"Download error for {url}: {e}\n")
        return False

# Function to extract specific videos from the ZIP file
def extract_video(zip_name):
    # Define the paths
    local_found = {}
    zip_file_path = os.path.join(zip_folder, zip_name)
    target_path = temp_video_folder

    # Run the unzip command to extract the specific video file
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for path in needed_paths[zip_name]:
                command = ['unzip','-j', zip_file_path, path, '-d', target_path]

                try:
                    subprocess.run(command, check=True)
                    print(f"✅ Extracted '{path}'")
                except subprocess.CalledProcessError as e:
                    print(f"An error occurred: {e}")
                final_idx = len(found) + 1  # Generate an index for the video
                final_video_path = os.path.join(temp_video_folder, f"a ({final_idx}).mp4")  # New name for the video
                final_prompt_path = os.path.join(temp_video_folder, f"a ({final_idx}).txt")
                if os.path.exists(final_video_path):
                    os.remove(final_video_path)  # Path for the caption text file
                os.rename(os.path.join(target_path,os.path.basename(path)), final_video_path)  # Rename the video
                with open(final_prompt_path, "w", encoding="utf-8") as f:
                    f.write(uuid_to_prompt[os.path.basename(path)])  # Write the caption for the video

                found[os.path.basename(path)] = True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to extract from {zip_file_path}")

# Function to download, extract videos, and save them
def download_and_extract(zip_name):
    local_found = {}
    zip_file_path = os.path.join(zip_folder, zip_name)

    # Check if the zip file already exists or needs to be downloaded
    if not os.path.exists(zip_file_path):  # If zip file is not already downloaded
        url = f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/{zip_name}"
        print(f"Downloading {zip_file_path}...")

        # Check for invalid ZIP file name or URL before downloading
        if not download_with_progress(url, zip_file_path):
            print(f"❌ Invalid ZIP file URL: {url}. Skipping download.")
            return local_found
    else:
        print(f"✅ Already downloaded: {zip_file_path}")

    # Extract relevant videos from the zip file
    try:
        extract_video(zip_name)
            # Remove the zip file to save space
        os.remove(zip_file_path)
        print(f"Deleted zip file: {zip_file_path}")

        return local_found

    except zipfile.BadZipFile as e:
        # Handle case if ZIP file is corrupted or invalid
        with open(error_log_path, "a") as log:
            log.write(f"Bad ZIP file error ({zip_file_path}): {e}\n")
        print(f"❌ Failed to open ZIP file: {zip_file_path}")
        return local_found


# Parallel execution with ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=3) as executor:  # Limiting to 3 workers for parallelism
    future_to_zip = {executor.submit(download_and_extract, zip_name): zip_name for zip_name in needed_zips}
    for future in tqdm(as_completed(future_to_zip), total=len(needed_zips), desc="Overall Progress"):
        result = future.result()

print(f"🎉 Completed. Extracted {len(found)} matching videos.")

print(found)

# Assuming merged_df contains the video_path and video columns
uuid_to_prompt = dict(zip(merged_df["video"], merged_df["caption"]))  # Map video UUID to prompt (caption)
uuid_set = set(merged_df["video"])  # Set of UUIDs we are interested in
found = {}  # Dictionary to track found videos

error_log_path = os.path.join(OUTPUT_DIRECTORY, "download_log.txt")  # Path for error log

zip_folder = os.path.join('/content',"zip_files")
temp_video_folder = os.path.join(OUTPUT_DIRECTORY, "dataset_train")
os.makedirs(zip_folder, exist_ok=True)
os.makedirs(temp_video_folder, exist_ok=True)

def extract_video(zip_name):
    # Define the paths
    local_found = {}
    zip_file_path = os.path.join(zip_folder, zip_name)
    target_path = temp_video_folder

    # Run the unzip command to extract the specific video file
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
          for path in needed_paths[zip_name]:
              command = ['unzip','-j', zip_file_path, path, '-d', target_path]

              try:
                  subprocess.run(command, check=True)
                  print(f"✅ Extracted '{path}'")
              except subprocess.CalledProcessError as e:
                  print(f"An error occurred: {e}")
              final_idx = len(found) + 1  # Generate an index for the video
              final_video_path = os.path.join(temp_video_folder, f"a ({final_idx}).mp4")  # New name for the video
              final_prompt_path = os.path.join(temp_video_folder, f"a ({final_idx}).txt")
              if os.path.exists(final_video_path):
                  os.remove(final_video_path)  # Path for the caption text file
              os.rename(os.path.join(target_path,os.path.basename(path)), final_video_path)  # Rename the video
              with open(final_prompt_path, "w", encoding="utf-8") as f:
                  f.write(uuid_to_prompt[os.path.basename(path)])  # Write the caption for the video

              found[os.path.basename(path)] = True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to extract from {zip_file_path}")

extract_video('OpenVid_part48.zip')

import subprocess

command = ['unzip','-j', '/content/finetuning-notebooks/openvid/zip_files/OpenVid_part48.zip', 'mnt/bn/videodataset-uswest/VDiT/dataset/panda-ours/4dCbHnmc1tk_14_0to658.mp4', '-d', 'target_folder']

try:
    subprocess.run(command, check=True)
    print("File extracted successfully.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")