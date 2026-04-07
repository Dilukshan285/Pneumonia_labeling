import os
import zipfile

output_dir = r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output"
zip_path = r"c:\Users\dviya\Desktop\Pneumonia_labeling\data\output_CSVs.zip"

print(f"Compressing heavy CSVs to bypass GitHub 100MB limit...")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)
                print(f"Zipped: {arcname} ({os.path.getsize(file_path) / 1024 / 1024:.1f} MB)")

print(f"Compression Complete! Saved as: {zip_path}")
