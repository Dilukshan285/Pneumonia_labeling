import concurrent.futures
import getpass
import io
import json
import os
import time
from collections import deque

import pandas as pd
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/"
TARGET_SIZE = (224, 224)

# Tuned defaults for high-throughput consumer hardware.
CPU_COUNT = os.cpu_count() or 8
DEFAULT_MAX_WORKERS = min(96, max(48, CPU_COUNT * 4))
MAX_WORKERS = int(os.getenv("PHYSIONET_MAX_WORKERS", str(DEFAULT_MAX_WORKERS)))
MAX_IN_FLIGHT = int(os.getenv("PHYSIONET_MAX_IN_FLIGHT", str(MAX_WORKERS * 2)))
MAX_RETRIES = int(os.getenv("PHYSIONET_MAX_RETRIES", "5"))
MAX_GLOBAL_ATTEMPTS = int(os.getenv("PHYSIONET_MAX_GLOBAL_ATTEMPTS", "40"))
CONNECT_TIMEOUT = float(os.getenv("PHYSIONET_CONNECT_TIMEOUT", "15"))
READ_TIMEOUT = float(os.getenv("PHYSIONET_READ_TIMEOUT", "45"))
WAIT_TIMEOUT = float(os.getenv("PHYSIONET_WAIT_TIMEOUT", "1"))
STALL_LOG_SECONDS = int(os.getenv("PHYSIONET_STALL_LOG_SECONDS", "15"))
# Retryable HTTP statuses where eventual success is likely.
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
# Save as uncompressed BMP to avoid additional lossy compression.
OUTPUT_FORMAT = "BMP"
OUTPUT_EXT = ".bmp"
RESAMPLE_METHOD = Image.Resampling.BILINEAR

SPLITS = {
    'train': 'data/output/pp1_train.csv',
    'val': 'data/output/pp1_val.csv',
    'test': 'data/output/pp1_test.csv'
}

OUTPUT_BASE_DIR = 'data/images'

CSV_COLUMNS = ['dicom_id', 'label', 'image_rel_path']


def create_session(username, password):
    """Create a tuned requests session for high parallel throughput."""
    session = requests.Session()
    session.auth = (username, password)
    session.headers.update({'User-Agent': 'Wget/1.21.2 (linux-gnu)'})

    retry_strategy = Retry(
        total=1,
        connect=1,
        read=1,
        status=1,
        backoff_factor=0.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS,
        max_retries=retry_strategy,
        pool_block=True,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def build_tasks(df, split_name):
    """Precompute all output paths/urls once to reduce per-worker overhead."""
    positive_dir = os.path.join(OUTPUT_BASE_DIR, split_name, 'positive')
    negative_dir = os.path.join(OUTPUT_BASE_DIR, split_name, 'negative')
    os.makedirs(positive_dir, exist_ok=True)
    os.makedirs(negative_dir, exist_ok=True)

    tasks = []
    already_present = 0

    for row in df.itertuples(index=False):
        dicom_id = str(row.dicom_id)
        label_value = int(float(row.label))
        rel_path = str(row.image_rel_path).replace('/ss', '/s')

        class_dir = positive_dir if label_value == 1 else negative_dir
        local_path = os.path.join(class_dir, f"{dicom_id}{OUTPUT_EXT}")

        if os.path.exists(local_path):
            already_present += 1
            continue

        tasks.append((dicom_id, f"{BASE_URL}{rel_path}", local_path))

    return tasks, already_present

# ==============================================================================
# DOWNLOAD AND RESIZE WORKER
# ==============================================================================
def download_and_resize(task, session, max_retries=MAX_RETRIES):
    """
    Downloads image from PhysioNet securely, resizes dynamically in memory,
    and handles retries silently.
    """
    dicom_id, url, local_path = task
    tmp_path = f"{local_path}.tmp"

    for attempt in range(max_retries):
        try:
            with session.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as resp:
                if resp.status_code == 200:
                    image_bytes = resp.content
                elif resp.status_code in (401, 403):
                    return "auth", f"Auth Error on {dicom_id}: {resp.status_code}. Bad username/password, or missing DUA."
                elif resp.status_code == 404:
                    return "not_found", f"Not Found on {dicom_id} ({url})"
                elif resp.status_code in RETRYABLE_STATUS_CODES:
                    if attempt == max_retries - 1:
                        return "retry", f"HTTP {resp.status_code} on {dicom_id}"
                    time.sleep(0.5 * (attempt + 1))
                    continue
                else:
                    return "fatal", f"HTTP {resp.status_code} on {dicom_id}"

            with Image.open(io.BytesIO(image_bytes)) as img:
                img = img.resize(TARGET_SIZE, RESAMPLE_METHOD)
                img.save(
                    tmp_path,
                    OUTPUT_FORMAT,
                )

            os.replace(tmp_path, local_path)
            return "ok", None

        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if attempt == max_retries - 1:
                return "retry", f"Network/Decode Error on {dicom_id}: {str(e)}"
            time.sleep(0.5 * (attempt + 1))


def run_split(split_name, csv_path, session):
    print(f"\n[{split_name.upper()}] Loading {csv_path}...")
    df = pd.read_csv(
        csv_path,
        usecols=CSV_COLUMNS,
        dtype={'dicom_id': str, 'image_rel_path': str},
        low_memory=False,
    )

    total_images = len(df)
    tasks, already_present = build_tasks(df, split_name)
    pending_images = len(tasks)

    print(
        f"[{split_name.upper()}] Total rows: {total_images:,} | "
        f"Already local: {already_present:,} | Pending download: {pending_images:,}"
    )

    if pending_images == 0:
        print(f"[{split_name.upper()}] All images already prepared.")
        return True

    print(
        f"[{split_name.upper()}] Starting with {MAX_WORKERS} workers "
        f"(in-flight cap: {MAX_IN_FLIGHT})..."
    )

    downloaded_count = 0
    hard_error_logs = []
    first_hard_error_printed = False
    first_auth_error_printed = False
    last_activity_ts = time.time()
    requeued_count = 0
    pending_queue = deque(tasks)
    attempt_counts = {task[0]: 0 for task in tasks}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {}

        def fill_pipeline():
            while pending_queue and len(future_to_task) < MAX_IN_FLIGHT:
                task = pending_queue.popleft()
                dicom_id = task[0]
                attempt_counts[dicom_id] += 1
                future = executor.submit(download_and_resize, task, session)
                future_to_task[future] = task

        fill_pipeline()

        with tqdm(total=pending_images, desc=f"{split_name.upper()} Pipeline", unit="img") as pbar:
            while future_to_task or pending_queue:
                done = set()
                if future_to_task:
                    done, _ = concurrent.futures.wait(
                        set(future_to_task.keys()),
                        timeout=WAIT_TIMEOUT,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                if not done:
                    now = time.time()
                    if now - last_activity_ts >= STALL_LOG_SECONDS:
                        completed = downloaded_count + len(hard_error_logs)
                        tqdm.write(
                            f"[{split_name.upper()}] Waiting on network... "
                            f"completed {completed}/{pending_images}, active {len(future_to_task)}, "
                            f"queued {len(pending_queue)}, requeued {requeued_count}"
                        )
                        last_activity_ts = now
                    fill_pipeline()
                    continue

                last_activity_ts = time.time()

                for future in done:
                    task = future_to_task.pop(future)
                    dicom_id = task[0]

                    try:
                        status, message = future.result()
                    except Exception as exc:
                        status, message = "retry", f"Worker failure on {dicom_id}: {exc}"

                    if status == "ok":
                        downloaded_count += 1
                        pbar.update(1)
                        continue

                    if status == "retry":
                        if attempt_counts[dicom_id] < MAX_GLOBAL_ATTEMPTS:
                            pending_queue.append(task)
                            requeued_count += 1
                        else:
                            final_msg = (
                                f"Retry limit reached on {dicom_id} after "
                                f"{attempt_counts[dicom_id]} attempts. Last error: {message}"
                            )
                            hard_error_logs.append(final_msg)
                            if not first_hard_error_printed:
                                tqdm.write(f"\nFIRST HARD ERROR: {final_msg}\n")
                                first_hard_error_printed = True
                            pbar.update(1)
                        continue

                    if status == "auth":
                        hard_error_logs.append(message)
                        if not first_auth_error_printed:
                            tqdm.write(f"\nAUTH BLOCKER: {message}\n")
                            first_auth_error_printed = True
                        executor.shutdown(wait=False, cancel_futures=True)
                        print("\nAuthentication failed. Aborting remaining splits.")
                        return False

                    hard_error_logs.append(message)
                    if not first_hard_error_printed:
                        tqdm.write(f"\nFIRST HARD ERROR: {message}\n")
                        first_hard_error_printed = True
                    pbar.update(1)

                fill_pipeline()

    success_count = already_present + downloaded_count
    print(f"[{split_name.upper()}] Success: {success_count}/{total_images} ready.")
    print(f"[{split_name.upper()}] Requeued transient failures: {requeued_count}")

    if hard_error_logs:
        print(f"[{split_name.upper()}] Encountered {len(hard_error_logs)} hard errors. Showing first 5:")
        for err in hard_error_logs[:5]:
            print(f"  - {err}")

    return len(hard_error_logs) == 0

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("=" * 70)
    print("  HIGH-PERFORMANCE PHYSIONET DOWNLOADER")
    print("=" * 70)
    print(
        f"Workers={MAX_WORKERS}, InFlight={MAX_IN_FLIGHT}, "
        f"Target={TARGET_SIZE[0]}x{TARGET_SIZE[1]}, Format={OUTPUT_FORMAT}, "
        f"GlobalAttempts={MAX_GLOBAL_ATTEMPTS}"
    )
    
    cred_file = "physionet_creds.json"
    
    if os.path.exists(cred_file):
        print("Loaded credentials from saved file.")
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
        print("Credentials saved locally for future runs.")
    
    print("\nBooting network session and worker pool...")
    session = create_session(username, password)

    # Process each split
    for split_name, csv_path in SPLITS.items():
        if not os.path.exists(csv_path):
            print(f"\nSkipping {split_name} (file not found: {csv_path})")
            continue

        should_continue = run_split(split_name, csv_path, session)
        if not should_continue:
            return

if __name__ == "__main__":
    main()
