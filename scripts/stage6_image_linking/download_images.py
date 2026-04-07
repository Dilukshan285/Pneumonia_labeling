import os
import io
import time
import getpass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from PIL import Image
import concurrent.futures
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/"
TARGET_SIZE = (224, 224)

# 🔥 Hardware Optimized for Ryzen 7 8845HS and 32GB RAM
# Network I/O is the bottleneck. 64 workers will massively saturate the bandwidth.
MAX_WORKERS = 64  

SPLITS = {
    'train': 'data/output/pp1_train.csv',
    'val': 'data/output/pp1_val.csv',
    'test': 'data/output/pp1_test.csv'
}

OUTPUT_BASE_DIR = 'data/images'

# ==============================================================================
# DOWNLOAD AND RESIZE WORKER
# ==============================================================================
def download_and_resize(row, split_name, session, max_retries=3):
    """
    Downloads image from PhysioNet securely, resizes dynamically in memory,
    and handles retries silently.
    """
    dicom_id = row['dicom_id']
    label = row['label']
    
    # 🐛 Fix the dataset typo where study IDs had double 's' (e.g., ss5098 -> s5098)
    rel_path = str(row['image_rel_path']).replace('/ss', '/s') 
    
    class_folder = 'positive' if label == 1 else 'negative'
    output_dir = os.path.join(OUTPUT_BASE_DIR, split_name, class_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    local_path = os.path.join(output_dir, f"{dicom_id}.jpg")
    
    if os.path.exists(local_path):
        return True
    
    url = f"{BASE_URL}{rel_path}"
    
    for attempt in range(max_retries):
        try:
            resp = session.get(url, stream=True, timeout=15)
            
            if resp.status_code == 200:
                with Image.open(io.BytesIO(resp.content)) as img:
                    img = img.convert('RGB')
                    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    img.save(local_path, "JPEG", quality=95)
                return True
            elif resp.status_code in [401, 403]:
                return f"Auth Error on {dicom_id}: {resp.status_code}. Bad username/password, or missing DUA."
            elif resp.status_code == 404:
                return f"Not Found on {dicom_id} ({url})"
            else:
                if attempt == max_retries - 1:
                    return f"HTTP {resp.status_code} on {dicom_id}"
                time.sleep(1) # Backoff
                
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Network Error on {dicom_id}: {str(e)}"
            time.sleep(1)

import json

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("=" * 70)
    print("  🚀 HIGH-PERFORMANCE PHYSIONET DOWNLOADER (Max Bandwidth)")
    print("=" * 70)
    
    cred_file = "physionet_creds.json"
    
    if os.path.exists(cred_file):
        print("✅ Automatically loaded credentials from saved file.")
        with open(cred_file, 'r') as f:
            creds = json.load(f)
            username = creds.get("username")
            password = creds.get("password")
    else:
        print("Please enter your PhysioNet credentials to access MIMIC-CXR.")
        print("Your password will be hidden while typing.\n")
        username = input("PhysioNet Username (or Email): ").strip()
        password = getpass.getpass("PhysioNet Password: ")
        
        # Save securely for next time
        with open(cred_file, 'w') as f:
            json.dump({"username": username, "password": password}, f)
        print("✅ Credentials saved locally for future runs!")
    
    print("\nVerifying credentials and booting thread pools...")
    
    # 🔥 Advanced Connection Pooling Strategy for High Concurrency
    # By default, Python requests only keeps 10 connections alive. 
    # Because we have 64 workers, 54 would get deadlocked. We increase the pool size!
    session = requests.Session()
    session.auth = (username, password)
    session.headers.update({'User-Agent': 'Wget/1.21.2 (linux-gnu)'})
    
    # Configure auto-retry and massive connection pools
    retry_strategy = Retry(
        total=5, status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # Process each split
    for split_name, csv_path in SPLITS.items():
        if not os.path.exists(csv_path):
            print(f"\n⚠️ Skipping {split_name} (File not found: {csv_path})")
            continue
            
        print(f"\n[{split_name.upper()}] Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        total_images = len(df)
        
        print(f"[{split_name.upper()}] Pumping {total_images:,} images with {MAX_WORKERS} workers...")
        
        success_count = 0
        error_logs = []
        first_error_printed = False
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(download_and_resize, row, split_name, session): row['dicom_id']
                for _, row in df.iterrows()
            }
            
            with tqdm(total=total_images, desc=f"{split_name.upper()} Pipeline", unit="img") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is True:
                        success_count += 1
                    else:
                        error_logs.append(result)
                        if not first_error_printed:
                            tqdm.write(f"\n❌ FIRST BLOCKER: {result}\n")
                            first_error_printed = True
                            if "Auth Error" in result:
                                print("\n❌ Authentication Failed! Aborting.")
                                return
                                
                    pbar.update(1)
        
        print(f"[{split_name.upper()}] ✅ Success: {success_count}/{total_images} downloaded.")
        if error_logs:
            print(f"[{split_name.upper()}] ⚠️ Encountered {len(error_logs)} errors. Showing first 5:")
            for err in error_logs[:5]:
                print(f"  - {err}")

if __name__ == "__main__":
    main()
