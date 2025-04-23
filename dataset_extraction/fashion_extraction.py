import os
import zipfile
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import subprocess
import pandas as pd


OUTPUT_DIRECTORY = "/content/drive/MyDrive/finetuning-notebooks/openvid"
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


# Keywords from SQL query
fashion_keywords = ['fashion']
context_keywords = ['advertisement']
brand_keywords = [
    'gucci', 'prada', 'louis vuitton', 'chanel', 'dior', 'versace',
    'balenciaga', 'givenchy', 'fendi', 'armani', 'dolce gabbana',
    'ysl', 'burberry'
]



def is_fashion_related(caption):
    if not isinstance(caption, str):
        return False
    caption_lower = caption.lower()
    return (
        ('fashion' in caption_lower and any(kw in caption_lower for kw in context_keywords)) or
        any(brand in caption_lower for brand in brand_keywords)
    )

# Apply filter
filtered_df = df[df['caption'].apply(is_fashion_related)]

# Select and rename relevant columns
filtered_df = filtered_df[[
    'video', 'caption', 'aesthetic score', 'motion score',
    'temporal consistency score', 'seconds'
]].rename(columns={'seconds': 'duration_seconds'})

# Sort by aesthetic_score
filtered_df = filtered_df.sort_values(by='aesthetic score', ascending=False)

print(f"Found {len(filtered_df)} fashion-related videos.")





uuid_to_prompt = dict(zip(filtered_df["video"], filtered_df["caption"]))
uuid_set = set(uuid_to_prompt.keys())
found = {}

error_log_path = os.path.join(OUTPUT_DIRECTORY, "download_log.txt")

zip_folder = "./zip_files"
temp_video_folder = os.path.join(OUTPUT_DIRECTORY, "data", "train")
os.makedirs(zip_folder, exist_ok=True)
os.makedirs(temp_video_folder, exist_ok=True)

def download_with_progress(url, local_path):
    try:
        response = requests.get(url, stream=True, timeout=300)
        total_size = int(response.headers.get('content-length', 0))
        with open(local_path, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(local_path)}",
            total=total_size, unit='B', unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))
        return local_path
    except Exception as e:
        with open(error_log_path, "a") as log:
            log.write(f"Download error ({url}): {e}\n")
        return None

def download_and_extract(i):
    local_found = {}
    url = f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{i}.zip"
    zip_file_path = os.path.join(zip_folder, f"OpenVid_part{i}.zip")

    if not os.path.exists(zip_file_path):
        if not download_with_progress(url, zip_file_path):
            return local_found
    else:
        print(f"✅ Already downloaded: {zip_file_path}")

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for member in tqdm(zip_ref.namelist(), desc=f"Extracting {zip_file_path}"):
                uuid = os.path.basename(member)
                if uuid in uuid_set and uuid not in local_found:
                    zip_ref.extract(member, temp_video_folder)
                    raw_path = os.path.join(temp_video_folder, member)
                    final_idx = len(found) + len(local_found) + 1
                    video_path = os.path.join(temp_video_folder, f"a ({final_idx}).mp4")
                    prompt_path = os.path.join(temp_video_folder, f"a ({final_idx}).txt")

                    os.rename(raw_path, video_path)
                    with open(prompt_path, "w", encoding="utf-8") as f:
                        f.write(uuid_to_prompt[uuid])

                    local_found[uuid] = True
    except zipfile.BadZipFile as e:
        with open(error_log_path, "a") as log:
            log.write(f"Zip error ({zip_file_path}): {e}\n")

    os.remove(zip_file_path)
    return local_found

# Parallel execution with ThreadPoolExecutor
tar_indices = range(31)
with ThreadPoolExecutor(max_workers=3) as executor:
    future_to_idx = {executor.submit(download_and_extract, idx): idx for idx in tar_indices}
    for future in tqdm(as_completed(future_to_idx), total=len(tar_indices), desc="Overall Progress"):
        result = future.result()
        found.update(result)

print(f"🎉 Completed. Extracted {len(found)} matching videos.")


