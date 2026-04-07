import os
import shutil

def organize_workspace():
    base_dir = r"c:\Users\dviya\Desktop\Pneumonia_labeling"
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(data_dir, "output")
    logs_dir = os.path.join(base_dir, "logs")
    scripts_dir = os.path.join(base_dir, "scripts")

    # Define target archive directories
    intermediates_dir = os.path.join(data_dir, "intermediates")
    archive_dir = os.path.join(output_dir, "deprecated_v1_archive")
    utils_dir = os.path.join(scripts_dir, "utils")

    # Create directories if they don't exist
    for d in [intermediates_dir, archive_dir, utils_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    print("="*60)
    print("ORGANIZING WORKSPACE (Archiving, Not Deleting)")
    print("="*60)

    # 1. Clean the main data/ directory (Move intermediate CSVs and stray logs)
    data_moved = 0
    for f in os.listdir(data_dir):
        src = os.path.join(data_dir, f)
        if os.path.isfile(src):
            if f.endswith('.txt'):
                shutil.move(src, os.path.join(logs_dir, f))
                data_moved += 1
            elif f.endswith('.csv'):
                shutil.move(src, os.path.join(intermediates_dir, f))
                data_moved += 1
    print(f"Moved {data_moved} intermediate files from /data to /data/intermediates")

    # 2. Clean the data/output/ directory (Keep only the 3 final deliverables)
    output_moved = 0
    keep_outputs = [
        'advanced_final_labels.csv', 
        'training_ready_labels.csv', 
        'final_image_training_manifest.csv',
        'advace_pne.zip'
    ]
    for f in os.listdir(output_dir):
        src = os.path.join(output_dir, f)
        # Move everything that isn't our final deliverable into the archive
        if os.path.isfile(src) and f not in keep_outputs:
            dest = os.path.join(archive_dir, f)
            # Handle potential overwriting if script was manually run twice
            if not os.path.exists(dest):
                shutil.move(src, dest)
            output_moved += 1
    print(f"Moved {output_moved} obsolete output files to /data/output/deprecated_v1_archive")

    # 3. Clean root Scripts (Move loose debugging/audit scripts to utils/)
    scripts_moved = 0
    loose_scripts = [
        'audit_labels.py', 'audit_labels_final.py', 'check_clean_neg.py', 
        'correct_labels.py', 'create_pos_neg_folders.py', 'extract_final_training_labels.py', 
        'extract_manual_verification_reports.py', 'extract_reports.py', 'extract_reports_corrected.py', 
        'refined_binary_labeling.py', 'verify_corrected.py', 'verify_final.py'
    ]
    for s in loose_scripts:
        src = os.path.join(scripts_dir, s)
        if os.path.exists(src):
            shutil.move(src, os.path.join(utils_dir, s))
            scripts_moved += 1
    print(f"Moved {scripts_moved} loose scripts to /scripts/utils/")

    print("\nWorkspace is now perfectly organized!")

if __name__ == "__main__":
    organize_workspace()
