"""
=======================================================================
GOD-TIER LIGHTNING-FAST GCP IMAGE DOWNLOADER v2 (ULTRA-SPEED)
=======================================================================
Optimized for Massive Throughput:
  - 256 Active Threads
  - Thread-local dedicated GCP Clients (no socket bottlenecks!)
  - In-memory 224x224 resize (GIL released during resize in Pillow)
=======================================================================
"""
import os
import io
import time
import threading
import pandas as pd
import concurrent.futures
from PIL import Image
from tqdm import tqdm
from google.cloud import storage

# ==============================================================================
# CONFIGURATION
# ==============================================================================
GCP_PROJECT_ID = "project-e92f175c-c161-4bd0-a0e"
BUCKET_NAME = "mimic-cxr-jpg-2.1.0.physionet.org"

# MASSIVE PARALLELISM
GCP_WORKERS = 64
TARGET_SIZE = (224, 224)
OUTPUT_FORMAT = "BMP"

CSVs = {
    'train': 'data/output/multi_label_dataset/ml_train.csv',
    'val':   'data/output/multi_label_dataset/ml_val.csv',
    'test':  'data/output/multi_label_dataset/ml_test.csv'
}

OUTPUT_BASE_DIR = 'data/output/multi_image'

# Thread-local storage to give EVERY thread its own dedicated network socket pool
thread_local = threading.local()

def get_thread_local_bucket():
    """Giving each thread its own GCP Client bypasses connection pool limits."""
    if not hasattr(thread_local, "bucket"):
        client = storage.Client(project=GCP_PROJECT_ID)
        thread_local.bucket = client.bucket(BUCKET_NAME, user_project=GCP_PROJECT_ID)
    return thread_local.bucket

# ==============================================================================
# CORE WORKER
# ==============================================================================
def download_and_resize(task, max_retries=5):
    dicom_id, blob_path, local_path = task
    bucket = get_thread_local_bucket()
    
    for attempt in range(max_retries):
        try:
            # 1. Download image bytes directly from bucket
            blob = bucket.blob(blob_path)
            image_bytes = blob.download_as_bytes(client=bucket.client)
            
            # 2. Resize in-memory (Pillow C-backend releases the GIL here)
            with Image.open(io.BytesIO(image_bytes)) as img:
                img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
                # 3. Save as BMP (fast, lossless)
                img.save(local_path, OUTPUT_FORMAT)
                
            return "ok", None
            
        except Exception as e:
            if attempt == max_retries - 1:
                return "error", f"Failed on {dicom_id}: {str(e)}"
            time.sleep(0.5 * (attempt + 1))

# ==============================================================================
# PIPELINE ORCHESTRATION
# ==============================================================================
def process_split(split, csv_path):
    print(f"\n{'='*70}\n  Processing {split.upper()} SPLIT\n{'='*70}")
    
    if not os.path.exists(csv_path):
        print(f"Skipping {split} (CSV not found: {csv_path})")
        return
        
    df = pd.read_csv(csv_path, low_memory=False)
    
    pos_dir = os.path.join(OUTPUT_BASE_DIR, split, 'positive')
    neg_dir = os.path.join(OUTPUT_BASE_DIR, split, 'negative')
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)
    
    tasks = []
    already_done = 0
    
    for row in df.itertuples(index=False):
        dicom_id = str(row.dicom_id)
        label_val = int(row.Pneumonia) # TARGET PRIMARY LABEL
        
        rel_path = str(row.image_rel_path).replace('/ss', '/s')
        blob_path = "files/" + rel_path 
        
        target_dir = pos_dir if label_val == 1 else neg_dir
        local_path = os.path.join(target_dir, f"{dicom_id}.bmp")
        
        if os.path.exists(local_path):
            already_done += 1
            continue
            
        tasks.append((dicom_id, blob_path, local_path))

    pending = len(tasks)
    print(f"Total: {len(df):,} | Already downloaded: {already_done:,} | Pending: {pending:,}")

    if pending == 0:
        return
        
    print(f"Launching {GCP_WORKERS} dedicated Google Cloud sockets...")
    
    completed = 0
    errors = 0
    error_msgs = []
    
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=GCP_WORKERS) as executor:
        # Submit without passing the global bucket explicitly
        future_to_task = {executor.submit(download_and_resize, t): t for t in tasks}
        
        with tqdm(total=pending, desc=f"{split.upper()} Downloads", unit="img") as pbar:
            for future in concurrent.futures.as_completed(future_to_task):
                status, msg = future.result()
                if status == "ok":
                    completed += 1
                else:
                    errors += 1
                    error_msgs.append(msg)
                
                pbar.update(1)
                
                if errors > 0:
                    pbar.set_postfix({"Errors": errors})

    total_time = time.time() - start_time
    print(f"\n{split.upper()} Complete! Downloaded {completed} images in {total_time/60:.2f} minutes.")
    
    if error_msgs:
        print(f"⚠️ {errors} Errors encountered. Showing first 5:")
        for msg in error_msgs[:5]:
            print(f"  - {msg}")

# ==============================================================================
# RUNNER
# ==============================================================================
if __name__ == "__main__":
    print("-" * 70)
    print("🚀 INITIALIZING ULTRA-SPEED GCP CONNECTION")
    print("-" * 70)
    
    # Try a quick ping to make sure auth works before lighting up 256 threads
    try:
        client = storage.Client(project=GCP_PROJECT_ID)
        client.bucket(BUCKET_NAME, user_project=GCP_PROJECT_ID)
    except Exception as e:
        print(f"❌ GCP ERROR: Could not connect. Reason: {e}")
        exit(1)
        
    for split_name, path in CSVs.items():
        process_split(split_name, path)
        
    print("\n🎉 ALL SPLITS COMPLETED SUCCESSFULLY! Dataset is ready for training.")
