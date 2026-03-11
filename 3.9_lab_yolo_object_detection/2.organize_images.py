#put all three image folder into one folder for the next step : label images

import os
import shutil
from pathlib import Path

# Paths
base_dir = Path(__file__).parent
source_folders = ['red_cup', 'blue_bottle', 'phone']  # your downloaded class folders
target_images_dir = base_dir / 'images'
target_images_dir.mkdir(exist_ok=True)

# Copy all images, renaming with class prefix to avoid duplicates
for class_name in source_folders:
    src_dir = base_dir / 'downloaded_images' / class_name
    if not src_dir.exists():
        print(f"Warning: {src_dir} does not exist")
        continue
    for img_file in src_dir.glob('*.[jJ][pP][gG]'):  # adjust extensions if needed
        new_name = f"{class_name}_{img_file.name}"
        dst = target_images_dir / new_name
        shutil.copy(img_file, dst)
        print(f"Copied {img_file.name} -> {new_name}")

print(f"All images copied to {target_images_dir}")

#run: python organize_images.py