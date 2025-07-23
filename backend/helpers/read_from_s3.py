# only taking 8 to see how well the finetuning is 

import boto3
import os

# --- Configuration ---
profile_name = 'personal'
bucket_name = "beatit-ai-training-data"
prefix = "training-data/"
search_keyword = "Drake Type Beat"
max_files = 8
download_dir = "./downloads"

# --- Setup ---
session = boto3.Session(profile_name=profile_name)
s3 = session.client('s3')

# Create download directory if not exists
os.makedirs(download_dir, exist_ok=True)

# --- Find matching files ---
paginator = s3.get_paginator('list_objects_v2')
page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

matching_files = []
for page in page_iterator:
    if 'Contents' in page:
        for obj in page['Contents']:
            key = obj['Key']
            if search_keyword in key:
                matching_files.append(key)
                if len(matching_files) == max_files:
                    break
    if len(matching_files) == max_files:
        break

# --- Download files ---
print("Downloading files:")
for key in matching_files:
    filename = os.path.basename(key)
    local_path = os.path.join(download_dir, filename)
    print(f"→ {key} → {local_path}")
    s3.download_file(bucket_name, key, local_path)

print("\n✅ Done.")
