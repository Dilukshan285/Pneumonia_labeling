import shutil
import os

source_dir = r"c:\Users\dviya\Downloads\mimic-cxr-reports\files"
target_zip = r"c:\Users\dviya\Desktop\Pneumonia_labeling\raw_mimic_reports"

print("Zipping 227,000 raw medical reports...")
shutil.make_archive(target_zip, 'zip', source_dir)
print(f"Success! Saved archive to: {target_zip}.zip")
